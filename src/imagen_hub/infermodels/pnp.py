import torch
import PIL
from imagen_hub.utils.save_image_helper import tensor_to_pil

class PNP():
    """
    A class for Plug-And-Play (Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation).
    
    References: https://github.com/MichalGeyer/pnp-diffusers
    """
    def __init__(self, device="cuda", sd_version="2.1"):
        """
        Initialize the PNP class.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            sd_version (str, optional): 1.5, 2.0 or 2.1. Defaults to "2.1".
        """
        from imagen_hub.pipelines.pnp.pnp import PNPPipeline
        self.pipe = PNPPipeline(sd_version=sd_version, device=device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42, num_inversion_steps=1000):
        """
        Perform inference on a source image based on provided prompts.
        Only target_prompt and src_image are needed.

        Args:
            src_image (PIL.Image): Source image.
            src_prompt (str, optional): Description or caption of the source image.
            target_prompt (str, optional): Desired description or caption for the output image.
            instruct_prompt (str, optional): Instruction prompt. Not utilized in this implementation.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            PIL.Image: Transformed image based on the provided prompts.
        """
        # Generate the transformed image as a tensor.
        tensor_image = self.pipe.generate(PIL_image=src_image, prompt=target_prompt, seed=seed, num_inversion_steps=num_inversion_steps)
        
        # Convert the tensor image back to PIL.Image.
        transformed_image = tensor_to_pil(tensor_image)
        
        # Resize the transformed image to match the source image size.
        src_size = src_image.size  # Get the size of the source image.
        resized_transformed_image = transformed_image.resize(src_size, PIL.Image.LANCZOS)
        
        return resized_transformed_image
