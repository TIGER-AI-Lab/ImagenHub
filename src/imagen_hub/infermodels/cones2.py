import torch
from PIL import Image

from imagen_hub.pipelines.cones2 import Cones2Pipeline

# WIP
class Cones2():
    def __init__(self, device="cuda"):
        self.device = device

    def infer_one_image(self,seed: int = 42):
        pass