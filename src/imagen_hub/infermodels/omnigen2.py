import torch
from torchvision.transforms.functional import to_tensor

from ..pipelines.omnigen2.omnigen2_src.omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from ..pipelines.omnigen2.omnigen2_src.omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
import accelerate

class OmniGen2:
    def __init__(self,weight="OmniGen2/OmniGen2",device=None,torch_dtype=torch.bfloat16):
        """
        Attributes:
            pipe (OmniGEn2Pipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "OmniGen2/OmniGen2".
        """
        if(device):
            self.device = device
        else:
            accelerator = accelerate.Accelerator()
            self.device = accelerator.device
        # Load pipeline
        self.pipeline = OmniGen2Pipeline.from_pretrained(
            weight,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Replace transformer
        self.pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            weight,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )

        # Move to device
        self.pipeline = self.pipeline.to(self.device, dtype=torch_dtype)
        

    def infer_one_image(
        self,
        prompt,
        negative_prompt=None,
        input_images=[],
        width=1024,
        height=1024,
        num_inference_steps=50,
        text_guidance_scale=5.0,
        image_guidance_scale=2.0,
        max_sequence_length=2048,
        seed=42
    ):

        self.negative_prompt = negative_prompt or (
            "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, "
            "mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, "
            "broken legs censor, censored, censor_bar"
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        results = self.pipeline(
            prompt=prompt,
            input_images=input_images,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        return results.images[0]