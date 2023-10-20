from diffusers import StableDiffusionInpaintPipeline
import torch
import os
import torchvision
from PIL import Image

def inpaint_text_gligen(pipe, prompt, background_path, bounding_box, gligen_phrase, config):
    images = pipe(
        prompt,
        num_images_per_prompt=1,
        gligen_phrases=[gligen_phrase],
        gligen_inpaint_image=Image.open(background_path).convert('RGB'),
        gligen_boxes=[[x / 512 for x in bounding_box]],
        gligen_scheduled_sampling_beta=config.gligen_scheduled_sampling_beta,
        output_type="numpy",
        num_inference_steps=config.num_inference_steps
    ).images
    return images

def select_inpainting_pipeline(name: str, device="cuda"):
    inpainting_pipe = None
    if name == "sd_inpaint":
        inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                ).to(device)
    elif name == "gligen_inpaint":
        try:
            # requires GLIGEN fork of diffusers
            from diffusers import StableDiffusionGLIGENPipeline
            inpainting_pipe = StableDiffusionGLIGENPipeline.from_pretrained(
                "gligen/diffusers-inpainting-text-box", torch_dtype=torch.float32
            ).to(device)
        except:
            raise ImportError("StableDiffusionGLIGENPipeline is not supported in . Use GLIGEN fork of diffusers instead.")
    return inpainting_pipe
