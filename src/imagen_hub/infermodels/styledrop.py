import torch
from PIL import Image
from imagen_hub.pipelines.styledrop import StyleDropPipeline

# WIP
class StyleDrop():
    def __init__(self, device="cuda") -> None:
        self.pipe = StyleDropPipeline(device)
        self.pipe.set_inference()
        self.style_weight = None

    def infer_one_image(self, object_in_image: str, style_of_image: str, seed: str = 42):
        out = self.pipe.process(prompt=object_in_image, 
                                adapter_postfix=style_of_image, 
                                adapter_path=self.style_weight
                                seed=seed)[0]
        return out