import torch
from PIL import Image

from imagen_hub.utils.image_helper import rgba_to_01_mask


class SDXL():
    """
    Stable Diffusion XL.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    """
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Attributes:
            pipe (StableDiffusionXLPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "stabilityai/stable-diffusion-xl-base-1.0".
        """
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.
        
        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.
            
        Returns:
            PIL.Image.Image: The inferred image.
        """

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator,
        ).images[0]
        return image

class SDXLInpaint():
    """
    Stable Diffusion XL for image inpainting tasks.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    """
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Attributes:
            pipe (StableDiffusionXLInpaintPipeline): The underlying inpainting pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for inpainting. Default is "stabilityai/stable-diffusion-xl-base-1.0".
        """
        from diffusers import StableDiffusionXLInpaintPipeline
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
        ).to(device)

    def infer_one_image(self, src_image: Image = None, local_mask_prompt: str = None, mask_image: Image = None, seed: int = 42):
        """
        Inpaints an image based on the given source image, local mask prompt, mask image, and seed.
        
        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Default is None.
            local_mask_prompt (str, optional): The caption for target image. Default is None.
            mask_image (PIL.Image.Image, optional): The mask image for inpainting. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.
        
        Returns:
            PIL.Image.Image: The inpainted image.
        """
        src_image = src_image.convert('RGB') # force it to RGB format

        # Check mask type
        if mask_image.mode == 'RGBA':
            mask_image = rgba_to_01_mask(mask_image, reverse=False, return_type="PIL")

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=local_mask_prompt,
            image=src_image,
            mask_image=mask_image,
            generator=generator,
        ).images[0]
        return image