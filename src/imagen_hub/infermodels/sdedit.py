import torch
import PIL

from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionImg2ImgPipeline

class SDEdit():
    """
    A wrapper class for StableDiffusionImg2ImgPipeline for image-to-image transformations based on prompts.

    Reference: https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/stable_diffusion/img2img
    """

    def __init__(self, device="cuda", weight="runwayml/stable-diffusion-v1-5"):
        """
        Attributes:
            pipe (StableDiffusionImg2ImgPipeline): The underlying image-to-image transformation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image-to-image transformations. Default is "runwayml/stable-diffusion-v1-5".
        """
        self.device = device
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            weight, torch_dtype=torch.float16)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.inverse_scheduler = DDIMInverseScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_vae_slicing()

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, strength=0.8, guidance_scale=7.5, seed=42):
        """
        Transform a given source image based on a target prompt with specified parameters.

        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Default is None.
            src_prompt (str, optional): The prompt for the source image. Default is None.
            target_prompt (str, optional): The prompt for the target image. Default is None.
            instruct_prompt (str, optional): Instructions for the image transformation. Default is None.
            strength (float, optional): Indicates how much to transform the reference image. Must be between 0 and 1. Default is 0.8.
            guidance_scale (float, optional): Encourages image generation closely linked to the text prompt, usually at the expense of lower image quality. Default is 7.5.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=target_prompt,
            image=src_image,
            generator=generator,
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        return image
