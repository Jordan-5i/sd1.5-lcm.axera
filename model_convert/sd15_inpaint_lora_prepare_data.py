from typing import List, Union
import numpy as np
import os
import tarfile
import onnxruntime
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image


# /opt/conda/envs/lcm-sd/lib/python3.9/site-packages/diffusers/loaders/textual_inversion.py:115
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
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-inpainting/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-inpainting/text_encoder",
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


# /opt/conda/envs/lcm-sd/lib/python3.9/site-packages/diffusers/schedulers/scheduling_lcm.py:222
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
    unet_session_main = onnxruntime.InferenceSession("output_onnx_inpaint/unet_sim_cut.onnx")
    time_input = np.load("output_onnx_inpaint/time_input.npy")
    
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-inpainting/vae")
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    os.makedirs("calib_data_unet_inpaint", exist_ok=True)
    os.makedirs("calib_data_vae_encoder_inpaint", exist_ok=True)
    os.makedirs("calib_data_vae_decoder_inpaint", exist_ok=True)

    calib_tarfile_unet = tarfile.open(f"calib_data_unet_inpaint/data.tar", "w")
    calib_tarfile_vae_encoder = tarfile.open(f"calib_data_vae_encoder_inpaint/data.tar", "w")
    calib_tarfile_vae_decoder = tarfile.open(f"calib_data_vae_decoder_inpaint/data.tar", "w")

    prompts = ["concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",]
    masks = [load_image("inpaint_samples/inpaint_mask.png")]
    images = [load_image("inpaint_samples/inpaint.png")]

    for p, (prompt, mask, image) in enumerate(zip(prompts, masks, images)):
        prompt_embeds_npy = get_embeds(prompt)
        prompt_name = prompt.replace(" ", "_")

        original_image = image
        init_image = image_processor.preprocess(image, height=512, width=512, crops_coords=None, resize_mode="default")
        init_image = init_image.to(dtype=torch.float32)

        latent = torch.randn([1, 4, 64, 64], generator=None, device="cpu", dtype=torch.float32, layout=torch.strided)
        mask_condition = mask_processor.preprocess(mask, height=512, width=512, crops_coords=None, resize_mode="default")
        masked_image = init_image * (mask_condition < 0.5)
        
        # prepare mask latents
        mask = torch.nn.functional.interpolate(mask_condition, size=(64, 64))
        mask = mask.to(dtype=torch.float32)
        masked_image = masked_image.to(dtype=torch.float32)
        encoder_output = vae.encode(masked_image)
        # /opt/conda/envs/lcm-sd/lib/python3.9/site-packages/diffusers/models/autoencoders/vae.py:793
        # encoder输出后，还需要做一次高斯采样，channel数由8->4
        masked_image_latents = encoder_output.latent_dist.sample(None)
        masked_image_latents = vae.config.scaling_factor * masked_image_latents
        masked_image_latents = masked_image_latents.to(dtype=torch.float32)

        calib_data = {}
        calib_data["x"] = masked_image.detach().numpy()
        np.save(f"calib_data_vae_encoder_inpaint/data_{p}.npy", calib_data)
        calib_tarfile_vae_encoder.add(f"calib_data_vae_encoder_inpaint/data_{p}.npy")

        print(p, prompt)
        for i, timestep in enumerate(timesteps):
            print(i, timestep)

            latent_model_input = latent
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            
            noise_pred = unet_session_main.run(None,
                                               {"sample": latent_model_input.detach().numpy(),
                                                "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input[i], axis=0),
                                                "encoder_hidden_states": prompt_embeds_npy})[0]

            calib_data = {}
            calib_data["sample"] = latent_model_input.detach().numpy()
            calib_data["/down_blocks.0/resnets.0/act_1/Mul_output_0"] = np.expand_dims(time_input[i], axis=0)
            calib_data["encoder_hidden_states"] = prompt_embeds_npy
            np.save(f"calib_data_unet_inpaint/data_{p}_{i}.npy", calib_data)
            calib_tarfile_unet.add(f"calib_data_unet_inpaint/data_{p}_{i}.npy")

            sample = latent
            model_output = noise_pred
            if i < 3:
                prev_timestep = timesteps[i + 1]
            else:
                prev_timestep = timestep

            # /opt/conda/envs/lcm-sd/lib/python3.9/site-packages/diffusers/schedulers/scheduling_lcm.py:541
            alpha_prod_t = alphas_cumprod[timestep]
            alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod

            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            # 3. Get scalings for boundary conditions
            # /opt/conda/envs/lcm-sd/lib/python3.9/site-packages/diffusers/schedulers/scheduling_lcm.py:490
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
        np.save(f"calib_data_vae_decoder_inpaint/data_{p}.npy", calib_data)
        calib_tarfile_vae_decoder.add(f"calib_data_vae_decoder_inpaint/data_{p}.npy")
        
    calib_tarfile_unet.close()
    calib_tarfile_vae_encoder.close()
    calib_tarfile_vae_decoder.close()


