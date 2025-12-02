import torch
import PIL

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


class RealEdit():
    """
    A wrapper class for StableDiffusionInstructPix2PixPipeline fine-tuned on the RealEdit dataset.
    
    RealEdit is designed to perform image editing based on natural language instructions.
    Reference: https://huggingface.co/peter-sushko/RealEdit
    Paper: https://arxiv.org/pdf/2502.03629
    """

    def __init__(self, device="cuda", weight="peter-sushko/RealEdit"):
        """
        Initialize the RealEdit model pipeline.
        
        Attributes:
            device (str): Device on which the pipeline runs.
            pipe (StableDiffusionInstructPix2PixPipeline): The main pipeline for instruction-based image editing.
        
        Args:
            device (str, optional): The device on which the pipeline should run. Defaults to "cuda".
            weight (str, optional): The pretrained model weights. Defaults to "peter-sushko/RealEdit".
        """
        self.device = device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, num_inference_steps: int = 50, image_guidance_scale: float = 2.0, seed: int = 42):
        """
        Edit a source image based on an instruction prompt.
        
        Args:
            src_image (PIL.Image.Image, optional): The source image in RGB format. Defaults to None.
            src_prompt (str, optional): The prompt for the source image (not used in this model). Defaults to None.
            target_prompt (str, optional): The prompt for the target image (not used in this model). Defaults to None.
            instruct_prompt (str, optional): Instruction for how to edit the image (e.g., "give him a crown"). Defaults to None.
            num_inference_steps (int, optional): Number of denoising steps. Higher values generally produce better quality but take longer. Defaults to 50.
            image_guidance_scale (float, optional): Guidance scale for image conditioning. Higher values make the output closer to the input image. Defaults to 2.0.
            seed (int, optional): Seed for random generator to ensure reproducibility. Defaults to 42.
        
        Returns:
            PIL.Image.Image: The edited image.
        """
        # Ensure the image is in RGB format
        src_image = src_image.convert('RGB')
        
        # Set up the generator for reproducibility
        generator = torch.manual_seed(seed)
        
        # Perform the image editing
        image = self.pipe(
            prompt=instruct_prompt,
            image=src_image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            generator=generator
        ).images[0]
        
        return image

