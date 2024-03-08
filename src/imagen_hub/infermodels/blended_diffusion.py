import torch
import PIL


class BlendedDiffusion():
    """
    A class for performing blended diffusion on images using the BlendedDiffusionPipeline.
    """
    def __init__(self, device="cuda"):
        """
        Initializes the BlendedDiffusion instance with the given device.
        It used all default hyperparameters in the original code

        Args:
            device (str): The device for running the pipeline ("cuda" or "cpu"). Default is "cuda".
        """
        from imagen_hub.pipelines.blended_diffusion import BlendedDiffusionPipeline
        self.pipe = BlendedDiffusionPipeline(device=device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, mask_image: PIL.Image.Image = None, local_mask_prompt: str = None, seed=42, iterations_num=8, skip_timesteps=25):
        """
        Inpaints an image based on the given source image, local mask prompt, mask image, and seed.

        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Default is None.
            local_mask_prompt (str, optional): The caption for target image. Default is None.
            mask_image (PIL.Image.Image, optional): The mask image for inpainting. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.
            iterations_num (int, optional): The number of iterations for the diffusion process. Default is 8.
            skip_timesteps (int, optional): The number of timesteps to skip. Default is 25.

        Returns:
            PIL.Image.Image: The inpainted image.
        """
        src_image = src_image.convert('RGB') # force it to RGB format

        self.pipe.set_seed(seed)
        image = self.pipe(
            src_image,
            mask_image,
            prompt=local_mask_prompt,
            iterations_num=iterations_num,
            skip_timesteps=skip_timesteps,
        )
        return image
