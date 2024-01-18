# Modified from https://github.com/MichalGeyer/pnp-diffusers/blob/main/preprocess.py

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import os
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import torchvision.transforms as T


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
            self.use_depth = True
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="fp16",
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')

        self.inversion_func = self.ddim_inversion

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True,
                                timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_path, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps

            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x

    @torch.no_grad()
    def extract_latents(self, num_steps, image_pil, save_path, timesteps_to_save,
                        inversion_prompt='', extract_reverse=False):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image_pil = T.Resize(512)(image_pil.convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        latent = self.encode_imgs(image)

        inverted_x = self.inversion_func(cond, latent, save_path, save_latents=not extract_reverse,
                                         timesteps_to_save=timesteps_to_save)
        latent_reconstruction = self.ddim_sample(inverted_x, cond, save_path, save_latents=extract_reverse,
                                                 timesteps_to_save=timesteps_to_save)
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction  # , latent_reconstruction
