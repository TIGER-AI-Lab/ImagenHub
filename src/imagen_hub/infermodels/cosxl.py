
from huggingface_hub import hf_hub_download
import torch
import PIL

class CosXL():
    """
    Cos Stable Diffusion XL 1.0 Base is tuned to use a Cosine-Continuous EDM VPred schedule.
    Reference: https://huggingface.co/stabilityai/cosxl
    """
    def __init__(self, device="cuda"):
        """
        Attributes:
            pipe (StableDiffusionXLPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
        """
        from diffusers import StableDiffusionXLPipeline
        from diffusers import EDMEulerScheduler
        from imagen_hub.pipelines.cosxl.utils import set_timesteps_patched

        EDMEulerScheduler.set_timesteps = set_timesteps_patched
        normal_file = hf_hub_download(repo_id="TIGER-Lab/cosxl", filename="cosxl.safetensors")
        self.pipe = StableDiffusionXLPipeline.from_single_file(normal_file, 
                                                               torch_dtype=torch.float16, 
                                                               safety_checker=None)
        self.pipe.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")
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
        image = self.pipe(prompt, 
                       negative_prompt="", 
                       guidance_scale=7, 
                       num_inference_steps=20,
                       generator=generator).images[0]
        return image

class CosXLEdit():
    """
    Edit Cos Stable Diffusion XL 1.0 Base is tuned to use a Cosine-Continuous EDM VPred schedule, and then upgraded to perform instructed image editing.
    Reference: https://huggingface.co/stabilityai/cosxl
    """
    def __init__(self, device="cuda"):
        """
        Attributes:
            pipe (CosStableDiffusionXLInstructPix2PixPipeline): The InstructPix2Pix pipeline for image transformation.

        Args:
            device (str, optional): Device on which the pipeline runs. Defaults to "cuda".
        """
        from diffusers import EDMEulerScheduler
        from imagen_hub.pipelines.cosxl.custom_pipeline import CosStableDiffusionXLInstructPix2PixPipeline
        from imagen_hub.pipelines.cosxl.utils import set_timesteps_patched

        EDMEulerScheduler.set_timesteps = set_timesteps_patched
        edit_file = hf_hub_download(repo_id="TIGER-Lab/cosxl", filename="cosxl_edit.safetensors")
        self.pipe = CosStableDiffusionXLInstructPix2PixPipeline.from_single_file(
            edit_file, num_in_channels=8
        )
        self.pipe.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction")
        self.device = device

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42):
        """
        Modifies the source image based on the provided instruction prompt.

        Args:
            src_image (PIL.Image.Image): Source image in RGB format.
            instruct_prompt (str): Caption for editing the image.
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)

        resolution = 1024
        preprocessed_image = src_image.resize((resolution, resolution))
        image = self.pipe(prompt=instruct_prompt,
                        image=preprocessed_image,
                        height=resolution,
                        width=resolution,
                        negative_prompt="", 
                        guidance_scale=7,
                        num_inference_steps=20,
                        generator=generator).images[0]
        image = image.resize((src_image.width, src_image.height))

        return image
