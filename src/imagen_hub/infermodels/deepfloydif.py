import torch
import PIL

from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
class DeepFloydIF():
    """
    DeepFloydIF for generating images based on a text prompt through a multi-stage diffusion process.

    The generation process is divided into three stages:
    - Stage 1: Initial encoding and generation using the "DeepFloyd/IF-I-XL-v1.0" model.
    - Stage 2: Refinement using the "DeepFloyd/IF-II-L-v1.0" model.
    - Stage 3: Upscaling (currently not functioning) using the "stabilityai/stable-diffusion-x4-upscaler" model.

    Each stage enables CPU offload for the model to manage memory usage effectively.

    Attributes:
        device (str): The device on which the model runs, default is "cuda".
        stage_1, stage_2, stage_3: Pipelines for each stage of the diffusion process.
        safety_modules (dict): Modules for feature extraction, safety checking, and watermarking.
    """
    def __init__(self, device="cuda"):
        self.device = device
        # stage 1
        self.stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0",
                                                         variant="fp16",
                                                         torch_dtype=torch.float16)
        self.stage_1.enable_model_cpu_offload()

        # stage 2
        self.stage_2 = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16
        )
        self.stage_2.enable_model_cpu_offload()

        # stage 3
        self.safety_modules = {
            "feature_extractor": self.stage_1.feature_extractor,
            "safety_checker": self.stage_1.safety_checker,
            "watermarker": self.stage_1.watermarker,
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", **self.safety_modules, torch_dtype=torch.float16
        )
        self.stage_3.enable_model_cpu_offload()


    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.stage_1.to(self.device)
        generator = torch.manual_seed(seed)
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)

        image = self.stage_1(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images
        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
        ).images

        # somehow isn't working
        #image = self.stage_3(prompt=prompt,
        #                     image=image,
        #                     noise_level=100,
        #                     generator=generator).images

        return pt_to_pil(image)[0]
