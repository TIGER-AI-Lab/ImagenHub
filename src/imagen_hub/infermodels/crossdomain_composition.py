import torch
from PIL import Image

from imagen_hub.pipelines.crossdomain_composition import CrossDomainCompositionPipeline

#WIP
class CrossDomainComposition():
    def __init__(self, device="cuda"):
        self.pipe = CrossDomainCompositionPipeline(device=device)

    def infer_one_image(self, init_image: Image = None, mask: Image = None, prompt: str = None):
        """
        param:
            src_image: file path
            mask_image: file path
            prompt : target image caption
        return:
            output image
        """
        image = self.pipe(
            init_img=init_image,
            mask=mask,
            prompt=prompt,
        )
        return image