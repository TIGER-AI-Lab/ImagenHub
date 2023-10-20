"""Self-implemented Text2Live pipeline for with source code and basic libraries like PIL, numpy, torch, etc."""

import os
from typing import List, Optional, Tuple, Union, Dict

from IPython.display import display
import PIL
from PIL import Image
import numpy as np
import random
import torch
import yaml
from tqdm import tqdm


from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF


# from diffusers

# from text2live_src
from .text2live_src.datasets.image_dataset import SingleImageDataset
from .text2live_src.models.clip_extractor import ClipExtractor
from .text2live_src.models.image_model import Model
from .text2live_src.util.losses import LossG
from .text2live_src.util.util import tensor2im, get_optimizer


import sys

# basic funcs
from imagen_hub.utils.image_helper import rgba_to_01_mask

def _preprocess_image(image: PIL.Image.Image, image_size):
    return image.convert("RGB")

class Text2LivePipeline():
    r"""
    Pipeline for Text2Live text-guided image generation (i.e., editing).

    Text2Live can *only* edit one image a time for its updating the model checkpoint during inference.

    This model is written from https://github.com/omerbt/Text2LIVE/blob/main/train_image.py
    Parameters:

    """
    def __init__(self,
                 device: str = None):
        # define model
        # get current path
        print("Loading models...")
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print("Using device:", self.device)
        current_path = os.path.dirname(os.path.abspath(__file__))
        # use default text2live image general models/setting config.
        with open(os.path.join(current_path, "./text2live_src/configs/image_config.yaml")) as f:
            config = yaml.safe_load(f)

        config['device'] = self.device
        self.config = config
        self.model = Model(self.config)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self,
                 original_image: PIL.Image.Image,
                 screen_text: str,
                 comp_text: str,
                 src_text: str,
                 epoch: int = 50,
                 ) -> Dict[str, PIL.Image.Image]:

        """ Text2Live edits one image only with several prompts.
        Parameters:
            original_image: The original image to be edited. RGB format.
            screen_text: The text describing the edit layer.
            comp_text: The text describing the full edited image.
            src_text: The text describing the input image.
            epoch: The number of epochs to train the model. 1000 in paper, but usually 50 is enough.
        Returns:
            A dictionary of PIL.Image.Image objects, with keys as the layer names (edited layer, full layer, etc.).
            The expected output is composite layer.
        """

        # create dataset, loader
        self.dataset = SingleImageDataset(self.config, original_image)
        # update text in config
        self.config['screen_text'] = screen_text
        self.config['comp_text'] = comp_text
        self.config['src_text'] = src_text

        if comp_text == src_text:
            if src_text.strip().endswith("."):
                self.config['comp_text'] = src_text + " Photo."
            else:
                self.config['comp_text'] = src_text + ". Photo."

        # define loss function
        self.clip_extractor = ClipExtractor(self.config)
        self.criterion = LossG(self.config, self.clip_extractor)

        # define optimizer, scheduler
        self.optimizer = get_optimizer(self.config, self.model.parameters())

        # run editing.
        for cur_epoch in tqdm(range(1, epoch + 1)):
            inputs = self.dataset[0]
            for key in inputs:
                if key != "step":
                    inputs[key] = inputs[key].to(self.config["device"])

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            for key in inputs:
                if key != "step":
                    inputs[key] = [inputs[key][0]]
            losses = self.criterion(outputs, inputs)
            loss_G = losses["loss"]
            log_data = losses
            log_data["epoch"] = cur_epoch

            # print("log_data:", log_data)
            # log current generated image to wandb
            # if cur_epoch % self.config["log_images_freq"] == 0:
            if cur_epoch == epoch:
                src_img = self.dataset.get_img().to(self.config["device"])
                with torch.no_grad():
                    output = self.model.render(self.model.netG(src_img), bg_image=src_img)
                for layer_name, layer_img in output.items():
                    image_numpy_output = tensor2im(layer_img)
                    log_data[layer_name] = image_numpy_output

            loss_G.backward()
            self.optimizer.step()

            # update learning rate
            if self.config["scheduler_policy"] == "exponential":
                self.optimizer.param_groups[0]["lr"] = max(self.config["min_lr"], self.config["gamma"] * self.optimizer.param_groups[0]["lr"])
            lr = self.optimizer.param_groups[0]["lr"]
            log_data["lr"] = lr

        # print("log_data:", log_data)
        return_images = {}
        for key in log_data.keys():
            # print(key, log_data[key])
            if key in ['edit', 'alpha', 'edit_on_greenscreen', 'composite']:
                return_images[key] = Image.fromarray(log_data[key])

        return return_images
