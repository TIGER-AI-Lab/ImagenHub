import torch
import PIL

class UniDiffuser():
    """
    A wrapper class for UniDiffuserPipeline to infer images based on prompts.

    References:
        https://huggingface.co/docs/diffusers/api/pipelines/unidiffuser

    Seems there's something wrong with the code. Doesn't work on Pytorch 1.X.
    """
    def __init__(self, device="cuda", weight="thu-ml/unidiffuser-v1"):
        """
        Attributes:
            pipe (UniDiffuserPipeline): The underlying UniDiffuser pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run, default is "cuda".
            weight (str, optional): The pre-trained model weights, default is "thu-ml/unidiffuser-v1".
        """
        from diffusers import UniDiffuserPipeline

        self.pipe = UniDiffuserPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
        )
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
            num_inference_steps=20,
            guidance_scale=8.0
        ).images[0]
        return image
