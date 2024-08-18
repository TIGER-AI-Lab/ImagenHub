import torch

class AuraFlow():
    """
    AuraFlow v0.2 is the fully open-sourced largest flow-based text-to-image generation model.
    Reference: https://huggingface.co/fal/AuraFlow
    """
    def __init__(self, device="cuda", weight="fal/AuraFlow-v0.2"):
        """
        Attributes:
            pipe (AuraFlowPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "fal/AuraFlow-v0.2".
        """
        from diffusers import AuraFlowPipeline
        self.pipe = AuraFlowPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16)
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
                        height=1024,
                        width=1024,
                        num_inference_steps=50, 
                        guidance_scale=3.5,
                        generator=generator).images[0]

        return image