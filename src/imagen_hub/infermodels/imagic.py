import torch
import PIL

from diffusers import DiffusionPipeline, DDIMScheduler


class Imagic():
    """
    A wrapper around the DiffusionPipeline for guided diffusion-based image transformation.

    Imagic is designed to modify images based on a provided target caption.
    Reference: https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb
    """
    def __init__(self, device="cuda", weight="CompVis/stable-diffusion-v1-4"):
        """
        Attributes:
            device (str): Device on which the pipeline runs.
            pipe (DiffusionPipeline): The main pipeline used for guided diffusion.
            cur_image (PIL.Image.Image): Current image being processed.
            cur_target_prompt (str): Current target prompt.

        Args:
            device (str, optional): The device on which the pipeline should run. Defaults to "cuda".
            weight (str, optional): The pretrained weights for the model. Defaults to "CompVis/stable-diffusion-v1-4".
        """
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(weight,
                                                      custom_pipeline="imagic_stable_diffusion",
                                                      safety_checker=None,
                                                      )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config)
        self.cur_image = None
        self.cur_target_prompt = None

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, alpha: float = 1, seed: int = 42):
        """
        Modify the source image based on the provided target caption.

        Args:
            src_image (PIL.Image.Image, optional): Source image in RGB format. If provided, it's converted to RGB. Defaults to None.
            target_prompt (str, optional): Caption for the target image. Defaults to None.
            alpha (float, optional): Strength of trained embedding. Values beyond 1 can result in stronger edits. Defaults to 1.
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.Generator(self.device).manual_seed(seed)

        if ((self.cur_image != src_image) and (self.cur_target_prompt != target_prompt)):
            # Fine tune the model
            self.pipe.train(target_prompt,
                            image=src_image,
                            generator=generator)
            self.cur_image = src_image
            self.cur_target_prompt = target_prompt

        image = self.pipe(alpha=alpha, guidance_scale=7.5,
                          num_inference_steps=50, generator=generator).images[0]

        return image
