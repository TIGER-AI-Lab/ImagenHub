from diffusers import AutoPipelineForText2Image
import torch


class Kandinsky():
    """
    Kandinsky 2.2 inherits best practices from Dall-E 2 and Latent diffusion while introducing some new ideas.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/kandinsky
    """
    def __init__(self, device="cuda", weight="kandinsky-community/kandinsky-2-2-decoder"):
        """
        Attributes:
            pipe (AutoPipelineForText2Image): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "kandinsky-community/kandinsky-2-2-decoder".
        """
        self.pipe = AutoPipelineForText2Image.from_pretrained(
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
                          prior_guidance_scale=1.0,
                          generator=generator).images[0]

        return image
