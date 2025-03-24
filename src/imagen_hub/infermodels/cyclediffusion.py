import torch
import PIL

class CycleDiffusion():
    """
    CycleDiffusion class to transform images based on text prompts using a diffusion model.
    Uses the CycleDiffusionPipeline from Huggingface Diffusers.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/cycle_diffusion
    """
    def __init__(self, device="cuda", weight="CompVis/stable-diffusion-v1-4"):
        """
        Initialize the CycleDiffusion class with a specified device and weight.

        Args:
            device (str): Device to load the model. Default is "cuda".
            weight (str): Pre-trained model weight. Default is "CompVis/stable-diffusion-v1-4".
        """
        from diffusers import CycleDiffusionPipeline, DDIMScheduler

        self.pipe = CycleDiffusionPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed=42):
        """
        Process an image using the diffusion model based on text prompts.

        Args:
            src_image (PIL.Image.Image, optional): Source image in RGB format. Default is None.
            src_prompt (str, optional): Caption for the source image. Default is None.
            target_prompt (str, optional): Caption for the target image. Default is None.
            instruct_prompt (str, optional): Instructional caption. Default is None. [Note: This argument is defined but not used in the method]
            seed (int, optional): Seed for randomness. Default is 42.

        Returns:
            PIL.Image.Image: Processed image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=target_prompt,
            source_prompt=src_prompt,
            image=src_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.8,
            guidance_scale=2,
            source_guidance_scale=1,
            generator=generator,
        ).images[0]
        return image
