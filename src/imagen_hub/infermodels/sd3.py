import torch

class SD3():
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-3-medium-diffusers"):
        """
        Attributes:
            pipe (StableDiffusion3Pipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "stabilityai/stable-diffusion-3-medium-diffusers".
        """
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(weight, torch_dtype=torch.float16).to(device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]
        return image
