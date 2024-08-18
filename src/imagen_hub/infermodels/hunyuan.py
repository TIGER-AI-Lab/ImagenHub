import torch

class HunyuanDiT():
    def __init__(self, device="cuda", weight="Tencent-Hunyuan/HunyuanDiT-Diffusers"):
        """
        Attributes:
            pipe (DiffusionPipeline): The underlying image generation pipeline object. Requires diffusers >= 0.28.1.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "Tencent-Hunyuan/HunyuanDiT-Diffusers". You can use "Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled" for distilled model (faster).
        """
        from diffusers import HunyuanDiTPipeline
        self.pipe = HunyuanDiTPipeline.from_pretrained(weight, torch_dtype=torch.float16)
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
            generator=generator,
        ).images[0]
        return image