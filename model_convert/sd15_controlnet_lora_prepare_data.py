from typing import List, Union
import numpy as np
import os
import tarfile
import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, AutoencoderKL
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
import cv2


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


def get_embeds(prompt = "Portrait of a pretty girl", ):
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5/text_encoder",
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

    timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()
    unet_session_main = onnxruntime.InferenceSession("output_onnx_controlnet/unet_sim_cut.onnx")
    time_input_unet = np.load("output_onnx_controlnet/time_input_unet.npy")
    time_input_controlnet = np.load("output_onnx_controlnet/time_input_controlnet.npy")
    
    os.makedirs("calib_data_unet_controlnet", exist_ok=True)
    os.makedirs("calib_data_controlnet", exist_ok=True)
    # os.makedirs("calib_data_vae_encoder_controlnet", exist_ok=True)
    os.makedirs("calib_data_vae_decoder_controlnet", exist_ok=True)
    
    calib_tarfile_unet = tarfile.open(f"calib_data_unet_controlnet/data.tar", "w")
    calib_tarfile_controlnet = tarfile.open(f"calib_data_controlnet/data.tar", "w")
    # calib_tarfile_vae_encoder = tarfile.open(f"calib_data_vae_encoder_controlnet/data.tar", "w")
    calib_tarfile_vae_decoder = tarfile.open(f"calib_data_vae_decoder_controlnet/data.tar", "w")
    
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5/vae")
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    control_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False)

    controlnet = onnxruntime.InferenceSession("output_onnx_controlnet/controlnet_sim_cut.onnx")
    vae_decoder = onnxruntime.InferenceSession("output_onnx_controlnet/sd15_vae_decoder_sim.onnx")
    
    guidance_scale = 1.5
    controlnet_conditioning_scale=0.8

    prompts = ["the mona lisa"]
    images = ["/wangjian/project/sd1.5-lcm.axera/input_image_vermeer.png"]
    for p, (prompt, image) in enumerate(zip(prompts, images)):
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

        prompt_embeds_npy = get_embeds(prompt)
        negative_prompt_embeds_npy = get_embeds("")
        prompt_embeds_npy = np.concatenate([negative_prompt_embeds_npy, prompt_embeds_npy])
        prompt_name = prompt.replace(" ", "_")
        latent = torch.randn([1, 4, 64, 64], generator=None, device="cpu", dtype=torch.float32, layout=torch.strided)

        print(p, prompt)
        for i, timestep in enumerate(timesteps):
            print(i, timestep)

            latent_model_input = torch.cat([latent] * 2)
            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds_npy
            
            # controlnet
            *down_block_res_samples, mid_block_res_sample = controlnet.run(None, 
                                                                        {"sample": control_model_input.detach().numpy(), \
                                                                        "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.stack([time_input_controlnet[i], time_input_controlnet[i]], axis=0), \
                                                                        "encoder_hidden_states": controlnet_prompt_embeds,
                                                                        "controlnet_cond": image.detach().numpy(),
                                                                        "conditioning_scale": np.array([0.8], dtype=np.float32)})

            down_block_additional_residuals_args = {f"down_block_additional_residuals_{_}": down_block_res_samples[_] for _ in range(12)}
            unet_input_feed = {"sample": latent_model_input.detach().numpy(), \
                                "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.stack([time_input_unet[i], time_input_unet[i]], axis=0), \
                                "encoder_hidden_states": prompt_embeds_npy, 
                                "mid_block_additional_residual": mid_block_res_sample}
            unet_input_feed.update(down_block_additional_residuals_args)
            noise_pred = unet_session_main.run(None, unet_input_feed)[0]

            # unet的输入
            calib_data = {}
            calib_data["sample"] = latent_model_input.detach().numpy()
            calib_data["/down_blocks.0/resnets.0/act_1/Mul_output_0"] = np.stack([time_input_unet[i], time_input_unet[i]], axis=0)
            calib_data["encoder_hidden_states"] = prompt_embeds_npy
            for _ in range(12):
                calib_data[f"down_block_additional_residuals_{_}"] = down_block_res_samples[_]
            calib_data["mid_block_additional_residual"] = mid_block_res_sample
            np.save(f"calib_data_unet_controlnet/data_{p}_{i}.npy", calib_data)
            calib_tarfile_unet.add(f"calib_data_unet_controlnet/data_{p}_{i}.npy")

            # controlnet的输入
            calib_data = {}
            calib_data["sample"] = control_model_input.detach().numpy()
            calib_data["/down_blocks.0/resnets.0/act_1/Mul_output_0"] = np.stack([time_input_controlnet[i], time_input_controlnet[i]], axis=0)
            calib_data["encoder_hidden_states"] = controlnet_prompt_embeds
            calib_data["controlnet_cond"] = image.detach().numpy()
            calib_data["conditioning_scale"] = np.array([0.8], dtype=np.float32)
            np.save(f"calib_data_controlnet/data_{p}_{i}.npy", calib_data)
            calib_tarfile_controlnet.add(f"calib_data_controlnet/data_{p}_{i}.npy")

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

        latent = latent / 0.18215
        calib_data = {}
        calib_data["x"] = latent.detach().numpy()
        np.save(f"calib_data_vae_decoder_controlnet/data_{p}.npy", calib_data)
        calib_tarfile_vae_decoder.add(f"calib_data_vae_decoder_controlnet/data_{p}.npy")
        
    calib_tarfile_unet.close()
    calib_tarfile_controlnet.close()
    calib_tarfile_vae_decoder.close()
