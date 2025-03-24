import torch
import PIL

class DiffEdit():
    """
    DiffEdit provides functionality to edit images using the StableDiffusionDiffEditPipeline
    from Hugging Face's diffusers.

    The pipeline makes use of two schedulers: DDIMScheduler and DDIMInverseScheduler.

    Reference: https://huggingface.co/docs/diffusers/api/pipelines/diffedit
    """
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-2-1"):
        """
        Attributes:
            pipe (StableDiffusionDiffEditPipeline): The main pipeline used for image editing.

        Args:
            device (str, optional): Device to load the pipeline on. Default is "cuda".
            weight (str, optional): Model weight to be loaded into the pipeline. Default is "stabilityai/stable-diffusion-2-1".
        """
        from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline
        self.device = device
        self.pipe = StableDiffusionDiffEditPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.inverse_scheduler = DDIMInverseScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42):
        """
        Generate an edited image based on source image and given prompts.

        Args:
            src_image (PIL.Image.Image): Source image to be edited.
            src_prompt (str, optional): Source image caption.
            target_prompt (str, optional): Target image caption.
            instruct_prompt (str, optional): Instruction prompt. Currently not used in this method.
            seed (int, optional): Random seed. Default is 42.

        Returns:
            PIL.Image.Image: Edited image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)
        mask_image = self.pipe.generate_mask(
            image=src_image,
            source_prompt=src_prompt,
            target_prompt=target_prompt,
            generator=generator
        )
        inv_latents = self.pipe.invert(
            prompt=src_prompt, image=src_image, generator=generator).latents
        image = self.pipe(
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            generator=generator,
            negative_prompt=src_prompt
        ).images[0]
        return image
