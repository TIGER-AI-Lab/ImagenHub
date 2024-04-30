import torch

class PixArtSigma:
    """
    PixArtSigma for T2I tasks
    Reference: https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma/blob/main/app.py
    """

    def __init__(self, device="cuda"):
        """
        Attributes:
            pipe (AutoPipelineForText2Image): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
        """
        from imagen_hub.pipelines.pixart.sigma.diffusers_patches import PixArtSigmaPipeline, pixart_sigma_init_patched_inputs
        from diffusers import Transformer2DModel

        print("Changing _init_patched_inputs method of diffusers.models.Transformer2DModel using diffusers_patches.pixart_sigma_init_patched_inputs")
        setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)

        transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", # No idea why PixArt-alpha/PixArt-Sigma-XL-2-1024-MS causes missing keys
            subfolder='transformer',
            torch_dtype=torch.float16,
        )
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
            transformer=transformer,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)

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
