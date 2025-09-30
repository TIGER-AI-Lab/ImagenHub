# icedit_pipeline.py

import os
import numpy as np
import torch
from PIL import Image
from diffusers import FluxFillPipeline


class ICEdit:
    def __init__(
        self,
        flux_path: str = "black-forest-labs/flux.1-fill-dev",
        lora_path: str = "RiverZ/normal-lora",
        device: str = "cuda",
        use_cpu_offload: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the ICEdit pipeline.

        Parameters:
            flux_path (str): Path to the base model.
            lora_path (str): Path to the LoRA weights.
            device (str): 'cuda' or 'cpu'.
            use_cpu_offload (bool): Enable CPU offloading.
            torch_dtype (torch.dtype): Precision to use.
        """
        self.pipe = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=torch_dtype)
        self.pipe.load_lora_weights(lora_path)

        if use_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)

    @torch.no_grad()
    def infer_one_image(
        self,
        prompt: str,
        src_image: Image.Image,
        seed: int = 42,
    ) -> Image.Image:
        """
        Edit the given image based on the instruction.

        Parameters:
            image (PIL.Image): The input image (RGB).
            instruction (str): Editing instruction.
            seed (int): Random seed for reproducibility.

        Returns:
            PIL.Image: The edited image.
        """
        if src_image.mode != "RGB":
            src_image = src_image.convert("RGB")

        if src_image.size[0] != 512:
            new_width = 512
            scale = new_width / src_image.size[0]
            new_height = int(src_image.size[1] * scale)
            new_height = (new_height // 8) * 8
            src_image = src_image.resize((new_width, new_height))
            print(f"[INFO] Resized image to {new_width}x{new_height}")

        instruction_full = (
            f"A diptych with two side-by-side images of the same scene. "
            f"On the right, the scene is exactly the same as on the left but {prompt}"
        )

        width, height = src_image.size
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(src_image, (0, 0))
        combined_image.paste(src_image, (width, 0))

        # Create mask for the right half
        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = 255
        mask = Image.fromarray(mask_array)

        generator = torch.Generator("cpu").manual_seed(seed)

        result = self.pipe(
            prompt=instruction_full,
            image=combined_image,
            mask_image=mask,
            height=height,
            width=width * 2,
            guidance_scale=50,
            num_inference_steps=28,
            generator=generator,
        ).images[0]

        result_cropped = result.crop((width, 0, width * 2, height))
        return result_cropped
