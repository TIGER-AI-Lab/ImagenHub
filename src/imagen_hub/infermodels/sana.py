import torch

class SANA:
    def __init__(self, device="cuda", weight="Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",pag_applied_layers="transformer_blocks.8"):
        """
        Attributes:
            pipe (SanaPAGPipelin): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers".
            pag_applied_layers (str, optional): The specific layers where the PAG (Parameter-Efficient Adaptation) technique is applied. Default is "transformer_blocks.8".
        """
        from diffusers import SanaPAGPipeline

        self.pipe = SanaPAGPipeline.from_pretrained(
            weight,
            torch_dtype=torch.bfloat16,
            pag_applied_layers=pag_applied_layers,
        )

        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)
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
            guidance_scale=5.0,
            pag_scale=2.0,
            num_inference_steps=20,
            generator=generator,
        )[0][0]

        return image
