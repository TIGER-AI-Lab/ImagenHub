from torchmetrics.multimodal.clip_score import CLIPScore
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


class MetricCLIPScore():
    def __init__(self, device="cuda") -> None:
        self.model = get_clipscore_model().to(device)
        self.device = device
    
    def evaluate(self, generated_image: Image.Image, prompt: str, normalize = True):
        """
        0.35 means very aligned most of the time, 0.25 is ok aligned
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        generated_image = transform(generated_image).unsqueeze(0).float().to(self.device)
        clip_score = self.model(generated_image, prompt).detach()
        
        return float(clip_score) / 100.0 if normalize else float(clip_score)

def get_clipscore_model():
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    return clip_score_fn