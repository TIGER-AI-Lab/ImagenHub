"""Self-implemented GLIDE pipeline for with source code and basic libraries like PIL, numpy, torch, etc."""

from typing import List, Optional, Tuple, Union

from IPython.display import display
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

# from diffusers
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from imagen_hub.utils.image_helper import rgba_to_01_mask

from .glide_text2im.download import load_checkpoint  # TODO, make it download online.
from .glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# from .utils import randn_tensor
# from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


def _preprocess_image(image: Union[List, PIL.Image.Image, torch.Tensor], image_size):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]
        # to torch

    if isinstance(image[0], PIL.Image.Image):
        w, h = image_size, image_size  # resize to 64x64 or 256x256
        image = [np.array(i.resize((w, h), resample=Image.BICUBIC))[None, :] for i in image]
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
        # print("mask is already a tensor")
        pass
    elif isinstance(mask, PIL.Image.Image):
        # print("mask is a PIL image")
        mask = [mask]

    if isinstance(mask[0], PIL.Image.Image):
        # print("mask[0] shape", mask[0].size)
        # print("mask[1] shape", mask[1].size)
        w, h = image_size, image_size  # resize to 64x64 or 256x256
        # resize
        mask = [m.resize((w, h), resample=PIL_INTERPOLATION["nearest"]) for m in mask]
        # for each mask, do the rgba_to_01_mask
        mask = [rgba_to_01_mask(m, reverse=True) for m in mask]
        # print("mask[0] shape", mask[0].shape)
        # print("mask[1] shape", mask[1].shape)
        mask = [m[np.newaxis, :] for m in mask]
        mask = np.concatenate(mask, axis=0)
        # print("mask shape after concatenation", mask.shape)
        mask = torch.from_numpy(mask)
        # if len(mask.shape) == 3:
        #     mask = mask.unsqueeze(1)
    elif isinstance(mask[0], torch.Tensor):
        # unseq_flag = True if len(mask) == 1 else False
        mask = torch.cat(mask, dim=0)
        # print("torch mask shape", mask.shape)
        # if unseq_flag:
        #     mask = mask.unsqueeze(0)

    return mask


class GlidePipeline():
    r"""
    Pipeline for GLIDE mask-guided image generation (i.e., editing).

    This model is written from https://github.com/openai/glide-text2im/blob/main/notebooks/inpaint.ipynb
    Parameters:

    """
    def __init__(self, device: str = None):
        # On GPU
        has_cuda = torch.cuda.is_available()
        if device is None:
            device = torch.device('cpu' if not has_cuda else 'cuda')
        else:
            device = torch.device(device)
        # Create base model.
        options = model_and_diffusion_defaults()

        options['inpaint'] = True
        options['use_fp16'] = has_cuda
        options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if has_cuda:
            model.convert_to_fp16()
        model.to(device)
        model.load_state_dict(load_checkpoint('base-inpaint', device))
        # print('total base parameters', sum(x.numel() for x in model.parameters()))
        print("Loaded base model.")

        # Create upsampler model for super-resolution. 64 -> 256
        options_up = model_and_diffusion_defaults_upsampler()
        options_up['inpaint'] = True
        options_up['use_fp16'] = has_cuda
        options_up['timestep_respacing'] = 'fast27'  # use 27 diffusion steps for very fast sampling
        model_up, diffusion_up = create_model_and_diffusion(**options_up)
        model_up.eval()
        if has_cuda:
            model_up.convert_to_fp16()
        model_up.to(device)
        model_up.load_state_dict(load_checkpoint('upsample-inpaint', device))
        # print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
        print("Loaded upsampler model.")

        # Save to class.
        self.options = options  # model config
        self.options_up = options_up  # upsampler config
        self.model = model
        self.diffusion = diffusion
        self.model_up = model_up
        self.diffusion_up = diffusion_up
        self.device = device

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def batch_tokenization(self, prompts: List[str], tokenizer):
        batch_size = len(prompts)
        batch_tokens, batch_mask = [], []  # token mask
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            batch_tokens.append(tokens)

        for i in range(batch_size):
            tokens, mask = tokenizer.padded_tokens_and_mask(
                            batch_tokens[i], self.options['text_ctx'])
            batch_tokens[i] = tokens
            batch_mask.append(mask)

        # Create the classifier-free guidance tokens (empty)
        # full_batch_size = batch_size * 2
        # batch_uncond_tokens, batch_uncond_mask = [], []
        uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )
        batch_uncond_tokens = [uncond_tokens] * batch_size
        batch_uncond_mask = [uncond_mask] * batch_size
        return batch_tokens, batch_mask, batch_uncond_tokens, batch_uncond_mask

    # Create an classifier-free guidance sampling function
    def model_fn(self, x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    @torch.no_grad()
    def __call__(
        self,
        original_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        prompts: List[str],
        guidance_scale: float = 5.0,
        upsample_temp: float = 0.997,
        # jump_length: int = 10,
        # jump_n_sample: int = 10,
        # generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        # return_dict: bool = True,
    ) -> List[PIL.Image.Image]:


        batch_size = len(prompts)

        self.guidance_scale = guidance_scale

        full_batch_size = batch_size * 2  # for additional classifier free guidance
        batch_tokens, batch_mask, batch_uncond_tokens, batch_uncond_mask = self.batch_tokenization(prompts, self.model.tokenizer)
        # Pack the tokens together into model kwargs.
        original_image_64 = _preprocess_image(original_image, self.options['image_size'])  # [batch, channel, 64, 64]
        mask_image_64 = _preprocess_mask(mask_image, self.options['image_size'])  # [batch, 1, 64, 64]
        original_image_256 = _preprocess_image(original_image, self.options_up['image_size'])  # [batch, channel, 256, 256]
        mask_image_256 = _preprocess_mask(mask_image, self.options_up['image_size'])  # [batch, 1, 256, 256]

        # print("original_image_64", original_image_64.shape, type(original_image_64))
        # print("mask_image_64", mask_image_64.shape, type(mask_image_64))
        # print("original_image_256", original_image_256.shape, type(original_image_256))
        # print("mask_image_256", mask_image_256.shape, type(mask_image_256))

        model_kwargs_base = dict(
            tokens=torch.tensor(
                batch_tokens + batch_uncond_tokens, device=self.device),
            mask=torch.tensor(
                batch_mask + batch_uncond_mask,
                dtype=torch.bool,
                device=self.device,
            ),
            inpaint_image=(original_image_64 * mask_image_64).repeat(2, 1, 1, 1).to(self.device),
            inpaint_mask=mask_image_64.repeat(2, 1, 1, 1).to(self.device),
        )

        print("Sampling...")
        # print("inpaint_image", model_kwargs_base['inpaint_image'].shape)
        # print("inpaint_mask", model_kwargs_base['inpaint_mask'].shape)

        def denoised_fn_base(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_start * (1 - model_kwargs_base['inpaint_mask'])
                + model_kwargs_base['inpaint_image'] * model_kwargs_base['inpaint_mask']
            )

        # Sample from the base model.
        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            self.model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs_base,
            cond_fn=None,
            denoised_fn=denoised_fn_base,
        )[:batch_size]
        self.model.del_cache()

        # Upsample the samples.
        batch_tokens, batch_mask, _, _ = self.batch_tokenization(prompts, self.model_up.tokenizer)
        # Create the model conditioning dict.
        model_kwargs_upsample = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=torch.tensor(
                batch_tokens, device=self.device
            ),
            mask=torch.tensor(
                batch_mask,
                dtype=torch.bool,
                device=self.device,
            ),

            # Masked inpainting image.
            # inpaint_image=(original_image_256 * mask_image_256).repeat(batch_size, 1, 1, 1).to(self.device),
            # inpaint_mask=mask_image_256.repeat(batch_size, 1, 1, 1).to(self.device),
            inpaint_image=(original_image_256 * mask_image_256).to(self.device),
            inpaint_mask=mask_image_256.to(self.device),
        )

        print("Upsampling...")
        # print("Batch_tokens", batch_tokens)
        # print("Batch_mask", batch_mask)
        # print("Batch_tokens", np.array(batch_tokens).shape)
        # print("Batch_mask", np.array(batch_mask).shape)
        # print("inpaint_image", model_kwargs_upsample['inpaint_image'].shape)
        # print("inpaint_mask", model_kwargs_upsample['inpaint_mask'].shape)

        def denoised_fn_upsample(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            # print("x_start", x_start.shape)
            # print("model_kwargs_upsample['inpaint_mask']", model_kwargs_upsample['inpaint_mask'].shape)
            # print("model_kwargs_upsample['inpaint_image']", model_kwargs_upsample['inpaint_image'].shape)
            return (
                x_start * (1 - model_kwargs_upsample['inpaint_mask'])
                + model_kwargs_upsample['inpaint_image'] * model_kwargs_upsample['inpaint_mask']
            )

        # Sample from the base model.
        self.model_up.del_cache()
        up_shape = (batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        up_samples = self.diffusion_up.p_sample_loop(
            self.model_up,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs_upsample,
            cond_fn=None,
            denoised_fn=denoised_fn_upsample,
        )[:batch_size]
        self.model_up.del_cache()

        scaled_up_samples = ((up_samples + 1)*127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        # transform each image to PIL image
        scaled_up_samples = scaled_up_samples.permute(0, 2, 3, 1).numpy()
        reshaped_up_samples = scaled_up_samples.reshape(batch_size, self.options_up["image_size"], self.options_up["image_size"], 3)

        up_images = [Image.fromarray(reshaped_up_samples[i]) for i in range(batch_size)]
        # up_images = Image.fromarray(reshaped_up_samples.numpy())

        return ImagePipelineOutput(images=up_images)
