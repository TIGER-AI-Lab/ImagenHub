#https://github.com/THUDM/ImageReward
import ImageReward as RM
from PIL import Image
from typing import List, Union
import os

class MetricImageReward():
    def __init__(self, device="cuda") -> None:
        self.model = get_imagereward_model()
        self.device = device

    def evaluate(self, generated_prompt, generated_image: Union[Image.Image, List, os.PathLike, str]):
        return evaluate_imagereward_score(generated_prompt, generated_image)

def get_imagereward_model():
    model = RM.load("ImageReward-v1.0")
    return model

def evaluate_imagereward_score(generated_prompt, generated_image: Union[Image.Image, List, os.PathLike, str], imagereward_model):
    #https://github.com/THUDM/ImageReward/blob/main/ImageReward/ImageReward.py
    rewards = imagereward_model.score(generated_prompt, generated_image)
    return rewards



