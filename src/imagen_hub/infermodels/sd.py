import torch
import PIL

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
from imagen_hub.utils.image_helper import rgba_to_01_mask

class SD():
    def __init__(self, device="cuda", weight="stabilityai/stable-diffusion-2-base"):
        """
        Attributes:
            pipe (DiffusionPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. Default is "stabilityai/stable-diffusion-2-base".
        """
        self.pipe = DiffusionPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.device = device
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

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

class SDInpaint():
    """
    A wrapper class for StableDiffusionInpaintPipeline to perform image inpainting tasks.
    """
    def __init__(self, device="cuda", weight="runwayml/stable-diffusion-inpainting"):
        """
        Attributes:
            pipe (StableDiffusionInpaintPipeline): The underlying inpainting pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for inpainting. Default is "runwayml/stable-diffusion-inpainting".
        """
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
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

class OpenJourney(SD):
    def __init__(self, device="cuda", weight="prompthero/openjourney"):
        """
        A class for the OpenJourney image generation model.

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for OpenJourney. Default is "prompthero/openjourney".
        """
        super().__init__(device=device, weight=weight)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        return super().infer_one_image(prompt, seed)

class LCM():
    def __init__(self, device="cuda", weight="SimianLuo/LCM_Dreamshaper_v7"):
        """
        A class for the Latent Consistency Model. Require diffusers >= 0.22
        Reference: https://github.com/luosiallen/latent-consistency-model#latent-consistency-models

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for OpenJourney. Default is "SimianLuo/LCM_Dreamshaper_v7".
        """
        self.pipe = DiffusionPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
        ).to(device)

    def infer_one_image(self, prompt: str = None, seed: int = 42, num_inference_steps: int = 4):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.
            num_inference_steps (int, optional): inference steps. Recommend: 1~8 steps. Paper used 4.

        Returns:
            PIL.Image.Image: The inferred image.
        
        Notes:
            num_inference_steps can be set to 1~50 steps. LCM support fast inference even <= 4 steps.
            (Max)I personally found 8 steps give a better result but the paper focus on using 4 steps.
        """
        generator = torch.manual_seed(seed)
        images = self.pipe(prompt=prompt, 
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=8.0, 
                            lcm_origin_steps=50, 
                            generator=generator,
                            output_type="pil").images
        return images[0]

class PlayGroundV2(SD):
    def __init__(self, device="cuda", weight="playgroundai/playground-v2-1024px-aesthetic"):
        """
        A class for the PlayGroundAI image generation model.

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for PlayGroundV2. Default is "playgroundai/playground-v2-1024px-aesthetic".
        """
        super().__init__(device=device, weight=weight)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed. It is recommend to use guidance_scale=3.0.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator, 
            guidance_scale=3.0,
        ).images[0]
        return image

class PlayGroundV2_5():
    def __init__(self, device="cuda", weight="playgroundai/playground-v2.5-1024px-aesthetic"):
        """
        A class for the PlayGroundAI image generation model. v2.5

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for PlayGroundV2. Default is "playgroundai/playground-v2.5-1024px-aesthetic".
        """
        self.pipe = DiffusionPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed. It is recommend to use guidance_scale=3.0.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator, 
            guidance_scale=3.0,
        ).images[0]
        return image
    
class StableCascade(SD):
    def __init__(self, device="cuda", weight="stabilityai/stable-cascade"):
        """
        A class for the stable cascade image generation model.
        Require a special version of diffusers 
        pip install git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887 --force

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "stabilityai/stable-cascade".
        """
        from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
        self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                                torch_dtype=torch.bfloat16).to(device)
        self.pipe = StableCascadeDecoderPipeline.from_pretrained(weight,
                                                                 torch_dtype=torch.float16).to(device)


    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed. It is recommend to use guidance_scale=3.0.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """

        prior_output = self.prior(
            prompt=prompt,
            height=1024,
            width=1024,
            negative_prompt='',
            guidance_scale=4.0,
            num_images_per_prompt=1,
            num_inference_steps=20
        )
        image = self.pipe(
            image_embeddings=prior_output.image_embeddings.to(torch.float16),
            prompt=prompt,
            negative_prompt='',
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=10
        ).images[0]
        return image
