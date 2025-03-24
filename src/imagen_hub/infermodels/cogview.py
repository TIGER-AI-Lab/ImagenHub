import torch
import PIL

class CogView3Plus():
    """
    CogView3-Plus-3B.
    Reference: https://huggingface.co/spaces/THUDM-HF-SPACE/CogView3-Plus-3B-Space/blob/main/app.py
    """
    def __init__(self, device="cuda", weight="THUDM/CogView3-Plus-3B"):
        """
        Attributes:
            pipe (CogView3PlusPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation.
        """
        from diffusers import CogView3PlusPipeline

        self.pipe = CogView3PlusPipeline.from_pretrained(weight, torch_dtype=torch.bfloat16)
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
            guidance_scale=7,
            num_images_per_prompt=1,
            num_inference_steps=50,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]

        return image
