import torch

class FLUX1schnell():
    """
    Timestep-distilled Flux is a series of text-to-image generation models based on diffusion transformers. 
    Reference: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
    """
    def __init__(self, device="cuda", weight="black-forest-labs/FLUX.1-schnell"):
        """
        Attributes:
            pipe (FluxPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "black-forest-labs/FLUX.1-schnell".
        """
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(weight, torch_dtype=torch.bfloat16)
        self.device = device

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.to(self.device)
        generator = torch.manual_seed(seed)
        image = self.pipe(prompt=prompt,
                        guidance_scale=0.,
                        height=1024,
                        width=1024,
                        num_inference_steps=4,
                        max_sequence_length=256,
                        generator=generator).images[0]

        return image

class FLUX1dev():
    """
    Guidance-distilled Flux is a series of text-to-image generation models based on diffusion transformers. 
    Reference: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
    """
    def __init__(self, device="cuda", weight="black-forest-labs/FLUX.1-dev"):
        """
        Attributes:
            pipe (FluxPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "black-forest-labs/FLUX.1-dev".
        """
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(weight, torch_dtype=torch.bfloat16)
        self.device = device

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.to(self.device)
        generator = torch.manual_seed(seed)
        image = self.pipe(prompt=prompt,
                        guidance_scale=3.5,
                        height=1024,
                        width=1024,
                        num_inference_steps=50,
                        generator=generator).images[0]

        return image