import torch
import PIL

class Glide():
    """
    A wrapper around the GlidePipeline for image inpainting.

    Glide is used to inpaint images based on provided source images, masks, and local prompts.
    Reference: https://github.com/openai/glide-text2im/

    Used all default hyperparameters in the original code
    """
    def __init__(self, device="cuda"):
        """
        Attributes:
            pipe (GlidePipeline): The main pipeline used for image inpainting.

        Args:
            device (str, optional): The device on which the pipeline should be loaded. Defaults to "cuda".
        """
        from imagen_hub.pipelines.glide import GlidePipeline
        self.pipe = GlidePipeline(device=device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, mask_image: PIL.Image.Image = None, local_mask_prompt: str = None, seed=42):
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
        self.pipe.set_seed(seed)
        image = self.pipe(
            src_image,
            mask_image,
            prompts=[local_mask_prompt],
        ).images[0]
        return image
