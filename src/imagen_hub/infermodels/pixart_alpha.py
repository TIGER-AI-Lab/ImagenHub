import torch

class PixArtAlpha():
    """
    PixArtAlpha for T2I tasks. Require diffusers >= 0.24.0 and transformers >=0.35.0
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/pixart
    """
    def __init__(self, device="cuda", weight="PixArt-alpha/PixArt-XL-2-1024-MS"):
        """
        Attributes:
            pipe (AutoPipelineForText2Image): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "PixArt-alpha/PixArt-XL-2-1024-MS".
        """
        from diffusers import PixArtAlphaPipeline

        self.pipe = PixArtAlphaPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
        ).to(device)
        self.device = device

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed. A single step is enough to generate high quality images.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.to(self.device)
        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator,
        ).images[0]
        return image
    
