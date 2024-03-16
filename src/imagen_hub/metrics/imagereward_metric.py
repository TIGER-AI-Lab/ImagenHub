
from PIL import Image
from typing import List, Union
import os

class MetricImageReward():
    """
    A wrapper for the ImageReward metric.

    This metric provides scores for generated images based on prompts.
    Reference: https://github.com/THUDM/ImageReward
    """
    def __init__(self, device="cuda") -> None:
        """
        Initializes the ImageReward metric wrapper.

        Args:
            device (str, optional): The device to load the model on. Defaults to "cuda".
        """
        self.model = get_imagereward_model()
        self.device = device

    def evaluate(self, generated_prompt, generated_image: Union[Image.Image, List, os.PathLike, str]):
        """
        Computes the ImageReward score for a generated image based on a given prompt.

        Args:
            generated_prompt (str): The prompt based on which the image was generated.
            generated_image (Union[Image.Image, List, os.PathLike, str]): The generated image or its path.

        Returns:
            float: The ImageReward score for the generated image.
        """
        return evaluate_imagereward_score(generated_prompt, generated_image)

def get_imagereward_model():
    """
    Loads and returns the pre-trained ImageReward model.

    Returns:
        Any: The ImageReward model.
    """
    #https://github.com/THUDM/ImageReward
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0")
    return model

def evaluate_imagereward_score(generated_prompt, generated_image: Union[Image.Image, List, os.PathLike, str], imagereward_model):
    """
    Computes the ImageReward score for a given generated image and prompt using a provided model.

    Args:
        generated_prompt (str): The prompt based on which the image was generated.
        generated_image (Union[Image.Image, List, os.PathLike, str]): The generated image or its path.
        imagereward_model (Any): The ImageReward model to use for the evaluation.

    Returns:
        float: The ImageReward score for the generated image.
    """    
    #https://github.com/THUDM/ImageReward/blob/main/ImageReward/ImageReward.py
    rewards = imagereward_model.score(generated_prompt, generated_image)
    return rewards
