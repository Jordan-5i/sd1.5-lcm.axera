from typing import List, Union
import numpy as np
import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, ControlNetModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
import time
import cv2
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="StableDiffusion",
        description="Generate picture with the input prompt"
    )
    parser.add_argument("--prompt", "-p", type=str, required=False, default="the mona lisa", help="the input text prompt")
    parser.add_argument("--image", "-i", type=str, required=False, default="../input_image_vermeer.png", help="the control image")
    parser.add_argument("--text_model_dir", "-e", type=str, required=False, default="../models/controlnet/", help="the dir of text encoder and tokenizer files")
    parser.add_argument("--unet_model", "-u", type=str, required=False, default="./output_onnx_controlnet/unet_sim_cut.onnx", help="the dir of unet.onnx")
    parser.add_argument("--controlnet_model", "-c", type=str, required=False, default="./output_onnx_controlnet/controlnet_sim_cut.onnx", help="the dir of unet.onnx")
    parser.add_argument("--vae_encoder_model", "-ve", type=str, required=False, default="./output_onnx_controlnet/sd15_vae_encoder_sim.onnx", help="the dir of vae_encoder.onnx")
    parser.add_argument("--vae_decoder_model", "-vd", type=str, required=False, default="./output_onnx_controlnet/sd15_vae_decoder_sim.onnx", help="the dir of vae_decoder.onnx")
    parser.add_argument("--time_input_unet", "-tu", type=str, required=False, default="../models/controlnet/time_input_unet.npy", help="the dir of time input file")
    parser.add_argument("--time_input_controlnet", "-tc", type=str, required=False, default="../models/controlnet/time_input_controlnet.npy", help="the dir of time input file")
    parser.add_argument("--save_dir", "-s", type=str, required=False, default="../lcm_lora_sdv1_5_controlnet_axmodel.png", help="the save dir of the output image")
    return parser.parse_args()


def maybe_convert_prompt(prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    if not isinstance(prompt, List):
        prompts = [prompt]
    else:
        prompts = prompt

    prompts = [_maybe_convert_prompt(p, tokenizer) for p in prompts]

    if not isinstance(prompt, List):
        return prompts[0]

    return prompts


def _maybe_convert_prompt(prompt: str, tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1

            prompt = prompt.replace(token, replacement)

    return prompt


def get_embeds(prompt = "Portrait of a pretty girl", tokenizer_dir = "../models/inpaint/tokenizer", text_encoder_dir = "../models/inpaint/text_encoder"):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir,
                                                 torch_dtype=torch.float32,
                                                 variant="fp16")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to("cpu"), attention_mask=None)

    prompt_embeds_npy = prompt_embeds[0].detach().numpy()
    return prompt_embeds_npy


def get_alphas_cumprod():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).detach().numpy()
    final_alphas_cumprod = alphas_cumprod[0]
    self_timesteps = np.arange(0, 1000)[::-1].copy().astype(np.int64)
    return alphas_cumprod, final_alphas_cumprod, self_timesteps


if __name__ == '__main__':
    args = get_args()
    prompt = args.prompt
    image = args.image
    tokenizer_dir = args.text_model_dir + 'tokenizer'
    text_encoder_dir = args.text_model_dir + 'text_encoder'
    unet_model = args.unet_model
    controlnet_model = args.controlnet_model
    vae_encoder_model = args.vae_encoder_model
    vae_decoder_model = args.vae_decoder_model
    time_input_unet = args.time_input_controlnet
    time_input_controlnet = args.time_input_controlnet
    save_dir = args.save_dir

    print(f"prompt: {prompt}")
    print(f"text_tokenizer: {tokenizer_dir}")
    print(f"text_encoder: {text_encoder_dir}")
    print(f"unet_model: {unet_model}")
    print(f"vae_encoder_model: {vae_encoder_model}")
    print(f"vae_decoder_model: {vae_decoder_model}")
    print(f"time_input_unet: {time_input_unet}")
    print(f"time_input_controlnet: {time_input_controlnet}")
    print(f"save_dir: {save_dir}")

    guidance_scale = 1.5
    controlnet_conditioning_scale=0.8

    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5/vae")
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    control_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False)
    
    image = load_image(image).resize((512, 512))
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    image = control_image_processor.preprocess(image, height=512, width=512).to(dtype=torch.float32)
    image = torch.cat([image] * 2)

    latent = torch.randn([1, 4, 64, 64], generator=None, device="cpu", dtype=torch.float32, layout=torch.strided)

    vae_encoder = onnxruntime.InferenceSession(vae_encoder_model)

    timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    
    # text encoder
    start = time.time()
    prompt_embeds_npy = get_embeds(prompt, tokenizer_dir, text_encoder_dir)
    negative_prompt_embeds_npy = get_embeds("", tokenizer_dir, text_encoder_dir)
    prompt_embeds_npy = np.concatenate([negative_prompt_embeds_npy, prompt_embeds_npy])
    print(f"text encoder take {1000 * (time.time() - start)}ms")
    prompt_name = prompt.replace(" ", "_")
    
    alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()
    
    # load unet model / vae model / controlnet model
    start = time.time()    
    unet_session_main = onnxruntime.InferenceSession(unet_model)
    controlnet = onnxruntime.InferenceSession(controlnet_model)
    vae_decoder = onnxruntime.InferenceSession(vae_decoder_model)
    print(f"load models take {1000 * (time.time() - start)}ms")
    
    # load time input file
    time_input_unet = np.load(time_input_unet)
    time_input_controlnet = np.load(time_input_controlnet)
    
    # unet inference loop
    unet_loop_start = time.time()    
    for i, timestep in enumerate(timesteps):
        # print(i, timestep)
        latent_model_input = torch.cat([latent] * 2)
        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds_npy

        # controlnet
        *down_block_res_samples, mid_block_res_sample = controlnet.run(None, 
                                                                    {"sample": control_model_input.detach().numpy(), \
                                                                    "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.stack([time_input_controlnet[i], time_input_controlnet[i]], axis=0), \
                                                                    "encoder_hidden_states": controlnet_prompt_embeds,
                                                                    "controlnet_cond": image.detach().numpy(),
                                                                    "conditioning_scale": np.array(0.8)})

        unet_start = time.time()
        down_block_additional_residuals_args = {f"down_block_additional_residuals_{_}": down_block_res_samples[_] for _ in range(12)}
        unet_input_feed = {"sample": latent_model_input.detach().numpy(), \
                            "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.stack([time_input_unet[i], time_input_unet[i]], axis=0), \
                            "encoder_hidden_states": prompt_embeds_npy, 
                            "mid_block_additional_residual": mid_block_res_sample}
        unet_input_feed.update(down_block_additional_residuals_args)
        noise_pred = unet_session_main.run(None, unet_input_feed)[0]
        print(f"unet once take {1000 * (time.time() - unet_start)}ms")

        if guidance_scale > 1:
            noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        sample = latent
        model_output = noise_pred
        if i < 3:
            prev_timestep = timesteps[i + 1]
        else:
            prev_timestep = timestep

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        scaled_timestep = timestep * 10
        c_skip = 0.5 ** 2 / (scaled_timestep ** 2 + 0.5 ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + 0.5 ** 2) ** 0.5
        predicted_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        denoised = c_out * predicted_original_sample + c_skip * sample

        if i != 3:
            noise = torch.randn(model_output.shape, generator=None, device="cpu", dtype=torch.float32,
                                layout=torch.strided).to("cpu").detach().numpy()
            prev_sample = (alpha_prod_t_prev ** 0.5) * denoised + (beta_prod_t_prev ** 0.5) * noise
        else:
            prev_sample = denoised

        latent = prev_sample

    print(f"unet loop take {1000 * (time.time() - unet_loop_start)}ms")

    # vae inference
    vae_start = time.time()    
    latent = latent / 0.18215
    image = vae_decoder.run(None, {"x": latent.detach().numpy()})[0]
    print(f"vae inference take {1000 * (time.time() - vae_start)}ms")
    
    # save result
    save_start = time.time() 
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])
    pil_image.save(save_dir)
    print(f"save image take {1000 * (time.time() - vae_start)}ms")