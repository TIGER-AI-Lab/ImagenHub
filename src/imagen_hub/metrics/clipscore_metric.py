import torch
from PIL import Image
import numpy as np
from torchvision import transforms


class MetricCLIPScore():
    """
    A class to compute the CLIPScore metric for evaluating the alignment between a generated image and a text prompt.
    """
    def __init__(self, device="cuda") -> None:
        """
        Initialize a MetricCLIPScore object with the specified device.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.model = get_clipscore_model().to(device)
        self.device = device

    def evaluate(self, generated_image: Image.Image, prompt: str, normalize = True):
        """
        Evaluate the alignment between the provided image and text prompt using the CLIPScore metric.
        
        Note: A score of 0.35 typically indicates strong alignment, while 0.25 is considered moderately aligned.
        
        Args:
            generated_image (Image.Image): The generated image for evaluation.
            prompt (str): The text prompt associated with the generated image.
            normalize (bool, optional): If True, normalize the score by dividing it by 100. Defaults to True.
            
        Returns:
            float: The computed CLIPScore.
        """
        generated_image = np.array(generated_image)
        generated_image = np.expand_dims(generated_image, axis=0)
        generated_image = (generated_image * 255).astype("uint8")
        generated_image = torch.from_numpy(generated_image).permute(0, 3, 1, 2).to(self.device)
        clip_score = self.model(generated_image, prompt).detach()
        return float(clip_score) / 100.0 if normalize else float(clip_score)

def get_clipscore_model():
    """
    Returns the CLIPScore model initialized with weights from "openai/clip-vit-base-patch16".
    
    Returns:
        torch.nn.Module: Initialized CLIPScore model.
    """
    from torchmetrics.multimodal.clip_score import CLIPScore
    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    return clip_score_fn
