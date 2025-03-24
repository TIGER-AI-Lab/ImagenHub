import torch

class SD3():
    def __init__(self, device="cuda", weight="v2ray/stable-diffusion-3-medium-diffusers", drop_encoder=True):
        """
        Attributes:
            pipe (StableDiffusion3Pipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "stabilityai/stable-diffusion-3-medium-diffusers".
            drop_encoder (bool, optional): Whether to drop the text encoder or not. Default is True. Significantly decrease the memory requirements for SD3 with only a slight loss in performance.
        """
        from diffusers import StableDiffusion3Pipeline
        if drop_encoder:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(weight, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16)
        else:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(weight, torch_dtype=torch.float16)
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
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]
        return image
