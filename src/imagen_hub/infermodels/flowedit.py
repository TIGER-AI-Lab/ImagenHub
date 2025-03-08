import torch
import PIL

from imagen_hub.pipelines.flowedit.FlowEdit_utils import FlowEditSD3, FlowEditFLUX
from diffusers import FluxPipeline, StableDiffusion3Pipeline

class FlowEdit_FLUX():
    """
    A wrapper class for FlowEdit image editing using the FLUX model.

    This class provides an interface to perform image editing tasks using the FLUX model.
    It supports both FLUX and SD3 model types for image-to-image transformations.

    Reference: https://huggingface.co/spaces/fallenshock/FlowEdit/blob/main/app.py
    """

    def __init__(self, device="cuda", weight="black-forest-labs/FLUX.1-dev", model_type="flux"):
        """
        Initializes the FlowEdit_FLUX class with the specified device, model weights, and model type.

        Attributes:
            pipe (StableDiffusionImg2ImgPipeline): The underlying image-to-image transformation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image-to-image transformations. Default is "black-forest-labs/FLUX.1-dev".
            model_type (str, optional): The type of model to use, either "flux" or "sd3". Default is "flux".
        """
        self.device = device
        self.model_type = model_type

        if self.model_type == "flux":
            self.pipe = FluxPipeline.from_pretrained(weight, torch_dtype=torch.float16)
        elif self.model_type == "sd3":
            self.pipe = StableDiffusion3Pipeline.from_pretrained(weight, torch_dtype=torch.float16)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        #self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_vae_slicing()

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed=42):
        """
        Transforms a given source image based on a target prompt with specified parameters.

        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Default is None.
            src_prompt (str, optional): The prompt for the source image. Default is None.
            target_prompt (str, optional): The prompt for the target image. Default is None.
            instruct_prompt (str, optional): Instructions for the image transformation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        self.pipe.to(self.device)
        image = src_image
        # crop image to have both dimensions divisible by 16 - avoids issues with resizing
        image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
        # We recommend 1024x1024 images for the best results. If the input images are too large, there may be out-of-memory errors.
        image = image.resize((1024, 1024))
        image_src = self.pipe.image_processor.preprocess(image)

        # cast image to half precision
        image_src = image_src.to(self.device).half()

        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
            x0_src = (x0_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            # send to cuda
            x0_src = x0_src.to(self.device)

        if self.model_type == "flux":
            x0_tar = FlowEditFLUX(self.pipe,
                                self.pipe.scheduler,
                                x0_src,
                                src_prompt,
                                target_prompt,
                                negative_prompt="",
                                T_steps=28, #total number of discretization steps.
                                n_avg=1,
                                src_guidance_scale=1.5,
                                tar_guidance_scale=5.5,
                                n_min=0,
                                n_max=23,)
        elif self.model_type == "sd3":
            x0_tar = FlowEditSD3(self.pipe,
                                self.pipe.scheduler,
                                x0_src,
                                src_prompt,
                                target_prompt,
                                negative_prompt="",
                                T_steps=50, #total number of discretization steps.
                                n_avg=1,
                                src_guidance_scale=3.5,
                                tar_guidance_scale=13.5,
                                n_min=0,
                                n_max=33,)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        x0_tar_denorm = (x0_tar / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            image_tar = self.pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        image_tar = self.pipe.image_processor.postprocess(image_tar)

        return image_tar[0]

class FlowEdit_SD3(FlowEdit_FLUX):
    """
    A wrapper class for FlowEdit image editing using the SD3 model.

    This class extends the FlowEdit_FLUX class to specifically use the SD3 model for image editing tasks.
    """
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-3-medium-diffusers", model_type="sd3"):
        super().__init__(device, weight, model_type)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None,  seed=42):
        """
        Transforms a given source image based on a target prompt with specified parameters using the SD3 model.

        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Default is None.
            src_prompt (str, optional): The prompt for the source image. Default is None.
            target_prompt (str, optional): The prompt for the target image. Default is None.
            instruct_prompt (str, optional): Instructions for the image transformation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        return super().infer_one_image(src_image, src_prompt, target_prompt, instruct_prompt, seed)