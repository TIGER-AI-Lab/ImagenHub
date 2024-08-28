import torch

class PixArtSigma:
    """
    PixArtSigma for T2I tasks
    Reference: https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma/blob/main/app.py <- Obsolete
    New Reference: https://huggingface.co/docs/diffusers/api/pipelines/pixart_sigma
    """

    def __init__(self, device="cuda"):
        """
        Attributes:
            pipe (AutoPipelineForText2Image): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
        """
        from diffusers import PixArtSigmaPipeline

        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        )
        self.pipe.enable_model_cpu_offload()
        self.device = device

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed. A single step is enough to generate high quality images.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:git
            PIL.Image.Image: The inferred image.
        """
        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator,
        ).images[0]
        return image
