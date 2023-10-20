import os
from PIL import Image
from diffusers import StableDiffusionPipeline
from .cones2_src.layout_guidance import layout_guidance_sampling

class Cones2Pipeline():
    def __init__(self, device="cuda", weight_path="") -> None:
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(weight_path).to(self.device)

    def inference(self, prompt, residual_dict, subject_list):
        image = layout_guidance_sampling(
                    seed=42,
                    device="cuda:0",
                    resolution=768,
                    pipeline=self.pipe,
                    prompt=prompt,
                    residual_dict=residual_dict,
                    subject_list=subject_list,
                    subject_color_dict=None,
                    layout=None, # layout disable
                    cfg_scale=7.5,
                    inference_steps=50,
                    guidance_steps=50,
                    guidance_weight=0.08,
                    weight_negative=-1e8,
                )

        return image