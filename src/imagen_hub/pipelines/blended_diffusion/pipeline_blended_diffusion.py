"""Self-implemented Blended Diffusion pipeline for with source code and basic libraries like PIL, numpy, torch, etc."""

import os
from typing import List, Optional, Tuple, Union

from IPython.display import display
import PIL
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF

# from diffusers

import sys

# blended_diffusion_src
from .blended_diffusion_src.CLIP import clip
from .blended_diffusion_src.optimization.losses import range_loss, d_clip_loss
from .blended_diffusion_src.optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR
from .blended_diffusion_src.utils.metrics_accumulator import MetricsAccumulator
# from .blended_diffusion_src.utils.video import save_video
from .blended_diffusion_src.utils.download import default_cache_dir, load_checkpoint

from .blended_diffusion_src.optimization.augmentations import ImageAugmentations
from .blended_diffusion_src.utils.visualization import show_tensor_image, show_editied_masked_image
from .blended_diffusion_src.guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# basic funcs
from imagen_hub.utils.image_helper import rgba_to_01_mask




def _preprocess_image(image: Union[List, PIL.Image.Image, torch.Tensor], image_size):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]
        # to torch

    if isinstance(image[0], PIL.Image.Image):
        w, h = image_size, image_size
        image = [np.array(i.resize((w, h), resample=Image.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)  # [batch, channel, height, width]
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)

    return image


def _preprocess_mask(mask: Union[List, PIL.Image.Image, torch.Tensor], image_size):
    if isinstance(mask, torch.Tensor):
        pass
    elif isinstance(mask, PIL.Image.Image):
        mask = [mask]

    if isinstance(mask[0], PIL.Image.Image):
        w, h = image_size, image_size  # resize to 64x64 or 256x256
        # resize
        mask = [m.resize((w, h), resample=Image.NEAREST) for m in mask]
        mask = [rgba_to_01_mask(m) for m in mask]
        mask = [m[np.newaxis, :] for m in mask]
        mask = np.concatenate(mask, axis=0)
        mask = torch.from_numpy(mask)
    elif isinstance(mask[0], torch.Tensor):
        mask = torch.cat(mask, dim=0)

    return mask


class BlendedDiffusionPipeline():
    r"""
    Pipeline for Blended Diffusion mask-guided image generation (i.e., editing).

    Blended Diffusion can *only* edit one image a time for its updating the model checkpoint during inference.

    This model is written from https://github.com/omriav/blended-diffusion/blob/master/optimization/image_editor.py
    Parameters:

    """
    def __init__(self,
                 model_output_size: int = 256,
                 timestep_respacing: str = "100",  # more for better diffusion results
                 device: str = None):
        # Set up Config
        self.model_config = model_and_diffusion_defaults()
        self.model_output_size = model_output_size
        self.timestep_respacing = timestep_respacing
        self.aug_num = 8  # used for CLIP augmentation

        # hard-coded for now
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.timestep_respacing,
                "image_size": self.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
        # Load models
        print("Loading models...")
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print("Using device:", self.device)
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)

        print("Loading diffusion...")
        if self.model_output_size == 256:
            self.model.load_state_dict(load_checkpoint("checkpoints/256_uncond", self.device))
        else:
            self.model.load_state_dict(load_checkpoint("checkpoints/512", self.device))

        self.model.eval().to(self.device)

        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        print("Loading CLIP...")
        self.clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False, download_root=os.path.join(default_cache_dir()))[0].eval()
        )
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )

        self.image_augmentations = ImageAugmentations(self.clip_size, self.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
        return unscaled_timestep

    def clip_loss(self, x_in, text_embed, batch_size):
        """
        Calculates the CLIP loss between the input image and the text prompt.
        """
        clip_loss = torch.tensor(0)

        if self.mask is not None:
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: batch_size].mean()

        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed)

        return dists.item()

    @torch.no_grad()
    def __call__(self,
                 original_image: Union[torch.FloatTensor, PIL.Image.Image],
                 mask_image: Union[torch.FloatTensor, PIL.Image.Image],
                 prompt: str,
                 batch_size: int = 4,
                 clip_guidance_lambda: float = 1000,
                 range_lambda: float = 50,
                 enforce_background: bool = True,
                 local_clip_guided_diffusion: bool = False,
                 ddim: bool = False,
                 iterations_num: int = 8,
                 skip_timesteps: int = 25,
                 ) -> PIL.Image.Image:

        """ Blended Diffusion edits one image only with a mask and prompt.
        Parameters:
            original_image: The original image to edit.
            mask_image: The mask to edit with.
            prompt: The prompt to edit with.
            batch_size: *For one image editing*, the number of edit candidate generations for CLIP ranking later.
            clip_guidance_lambda: Controls how much the image should look like the prompt.
            range_lambda: Controls how far out of range RGB values are allowed to be.
        Returns:
            The edited image.

        """
        print("self.device:", self.device)
        text_embed = self.clip_model.encode_text(
            clip.tokenize(prompt).to(self.device)
        ).float()
        print("text_embed device:", text_embed.device)
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image = _preprocess_image(original_image, self.model_config["image_size"]).to(self.device)

        if mask_image is not None:
            self.mask = _preprocess_mask(mask_image, self.model_config["image_size"]).to(self.device)
        else:
            self.mask = torch.ones_like(self.init_image, device=self.device)

        def cond_fn(x, t, y=None):
            if prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)
                if clip_guidance_lambda != 0:
                    clip_loss = self.clip_loss(x_in, text_embed, batch_size) * clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())

                if range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                # default false, remove this hyper-param
                # if self.args.background_preservation_loss:
                #     if self.mask is not None:
                #         masked_background = x_in * (1 - self.mask)
                #     else:
                #         masked_background = x_in

                #     if self.args.lpips_sim_lambda:
                #         loss = (
                #             loss
                #             + self.lpips_model(masked_background, self.init_image).sum()
                #             * self.args.lpips_sim_lambda
                #         )
                #     if self.args.l2_sim_lambda:
                #         loss = (
                #             loss
                #             + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                #         )

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out

        # save_image_interval = self.diffusion.num_timesteps // 5
        for iteration_number in range(iterations_num):
            print(f"Start iterations {iteration_number}")

            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.model_output_size == 256
                else {
                    "y": torch.zeros([batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None if local_clip_guided_diffusion else postprocess_fn,
                randomize_class=True,
            )

            total_steps = self.diffusion.num_timesteps - skip_timesteps - 1
            min_distance = 1e5  # select the best image with the smallest distance to the prompt
            for j, sample in enumerate(samples):

                for b in range(batch_size):
                    pred_image = sample["pred_xstart"][b]

                    if (
                        self.mask is not None
                        and enforce_background
                        and j == total_steps
                        and not local_clip_guided_diffusion
                    ):
                        pred_image = (
                            self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                        )
                    pred_image = pred_image.add(1).div(2).clamp(0, 1)
                    pred_image_pil = TF.to_pil_image(pred_image)
                    masked_pred_image = self.mask * pred_image.unsqueeze(0)
                    final_distance = self.unaugmented_clip_distance(
                        masked_pred_image, text_embed
                    )

                    if final_distance < min_distance:
                        min_distance = final_distance
                        best_image = pred_image_pil

        return best_image
