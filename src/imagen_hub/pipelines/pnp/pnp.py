# Modified from https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp.py
# Support diffusers 0.21+

import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .pnp_utils import seed_everything, register_time, load_source_latents_t, register_attention_control_efficient, register_conv_control_efficient
from .preprocess import Preprocess, get_timesteps

# suppress partial model loading warning
logging.set_verbosity_error()

class PNPPipeline(nn.Module):
    def __init__(self, sd_version="2.1", steps=50, guidance_scale=7.5, device="cuda", LOW_MEMORY=False):
        super().__init__()
        self.device = device
        self.steps = steps
        self.sd_version = sd_version
        self.guidance_scale = guidance_scale

        if sd_version == '2.1':
            self.model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            self.model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            self.model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        pipe = StableDiffusionPipeline.from_pretrained(self.model_key, torch_dtype=torch.float16).to(self.device)
        if LOW_MEMORY:
            pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.pt_path = None

        self.scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(self.steps, device=self.device)

    def generate(self, PIL_image, prompt, negative_prompt="", pnp_f_t=0.8, pnp_attn_t=0.5, seed=42, pt_path=os.path.join("temp", "pnp", f"latents"), num_inversion_steps=1000):
        seed_everything(seed)
        self.pt_path = pt_path
        
        #clear out old latents
        for f in glob.glob(os.path.join(pt_path, "noisy_latents_*.pt")):
            os.remove(f)
        
        self.image, self.eps = self.set_image(PIL_image, pt_path, num_inversion_steps=num_inversion_steps)

        self.text_embeds = self.get_text_embeds(prompt, negative_prompt)
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

        pnp_f_t = int(self.steps * pnp_f_t)
        pnp_attn_t = int(self.steps * pnp_attn_t)
        
        # Override the default UNet transformers blocks with the ones that support PnP
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)

        # Reverse diffusion
        edited_img = self.sample_loop(self.eps)
        return edited_img

    def set_image(self, PIL_image, pt_path, num_inversion_steps=1000):
        os.makedirs(pt_path, exist_ok=True)
        # do inversion
        toy_scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        toy_scheduler.set_timesteps(1000)
        timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, 
                                                                num_inference_steps=1000,
                                                                strength=1.0,
                                                                device=self.device)
        del toy_scheduler
        prep_model = Preprocess(self.device, sd_version=self.sd_version, hf_key=None)
        recon_image = prep_model.extract_latents(image_pil=PIL_image,
                                            num_steps=num_inversion_steps-1,
                                            save_path=pt_path,
                                            timesteps_to_save=timesteps_to_save,
                                            inversion_prompt="",
                                            extract_reverse=False)
        del prep_model
        T.ToPILImage()(recon_image[0]).save(os.path.join(pt_path, f'recon.jpg'))

        # get noise
        image = PIL_image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        latents_path = os.path.join(pt_path, f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        return image, noisy_latent

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.no_grad()
    def denoise_step(self, x, t):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, self.pt_path)
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def sample_loop(self, x, decoded=True):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            if decoded:
                x = self.decode_latent(x)
        return x
