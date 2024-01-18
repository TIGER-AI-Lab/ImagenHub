import torch
import PIL

from imagen_hub.pipelines.pnp.pnp import PNPPipeline
from imagen_hub.utils.save_image_helper import tensor_to_pil

class PNP():
    def __init__(self, device="cuda", sd_version="2.1"):
        self.pipe = PNPPipeline(sd_version=sd_version, device=device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42):
        tensor_image = self.pipe.generate(PIL_image=src_image, prompt=target_prompt, seed=seed)
        return tensor_to_pil(tensor_image)
