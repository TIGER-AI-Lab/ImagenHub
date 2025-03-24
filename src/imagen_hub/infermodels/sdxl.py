import torch
import PIL

from imagen_hub.utils.image_helper import rgba_to_01_mask


class SDXL():
    """
    Stable Diffusion XL.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    """
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Attributes:
            pipe (StableDiffusionXLPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "stabilityai/stable-diffusion-xl-base-1.0".
        """
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
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
        ).images[0]
        return image

class SDXLLightning():
    """
    SDXL-Lightning.
    Reference: https://huggingface.co/ByteDance/SDXL-Lightning
    """
    def __init__(self, device="cuda", weight="ByteDance/SDXL-Lightning"):
        """
        Attributes:
            pipe (StableDiffusionXLPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "ByteDance/SDXL-Lightning".
        """
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        ckpt = "sdxl_lightning_4step_unet.safetensors"
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(weight, ckpt), device="cuda"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

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
        image = self.pipe(prompt, num_inference_steps=4, guidance_scale=0, generator=generator).images[0]
        return image

class SDXLInpaint():
    """
    Stable Diffusion XL for image inpainting tasks.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    """
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Attributes:
            pipe (StableDiffusionXLInpaintPipeline): The underlying inpainting pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for inpainting. Default is "stabilityai/stable-diffusion-xl-base-1.0".
        """
        from diffusers import StableDiffusionXLInpaintPipeline
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
        ).to(device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, local_mask_prompt: str = None, mask_image: PIL.Image.Image = None, seed: int = 42):
        """
        Inpaints an image based on the given source image, local mask prompt, mask image, and seed.

        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Default is None.
            local_mask_prompt (str, optional): The caption for target image. Default is None.
            mask_image (PIL.Image.Image, optional): The mask image for inpainting. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inpainted image.
        """
        src_image = src_image.convert('RGB') # force it to RGB format

        # Check mask type
        if mask_image.mode == 'RGBA':
            mask_image = rgba_to_01_mask(mask_image, reverse=False, return_type="PIL")

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=local_mask_prompt,
            image=src_image,
            mask_image=mask_image,
            generator=generator,
        ).images[0]
        return image

class SDXLTurbo():
    """
    Stable Diffusion XL Turbo for T2I tasks. Require diffusers >= 0.24.0
    Reference: https://huggingface.co/stabilityai/sdxl-turbo
    """
    def __init__(self, device="cuda", weight="stabilityai/sdxl-turbo"):
        """
        Attributes:
            pipe (AutoPipelineForText2Image): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "https://huggingface.co/stabilityai/sdxl-turbo".
        """
        from diffusers import AutoPipelineForText2Image

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed. 1 step is enough to generate high quality images. 4 steps are claimed to be better but 1 step is the selling point.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=1, 
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        return image

class SSD(SDXL):
    def __init__(self, device="cuda", weight="segmind/SSD-1B"):
        """
        A class for the Segmind Stable Diffusion image generation model.
        Reference: https://huggingface.co/segmind/SSD-1B

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for SSD. Default is "segmind/SSD-1B".
        """
        super().__init__(device=device, weight=weight)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        return super().infer_one_image(prompt, seed)
