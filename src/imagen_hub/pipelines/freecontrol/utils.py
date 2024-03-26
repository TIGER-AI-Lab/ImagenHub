import hashlib
import io
import os
import re

import PIL.Image
import PIL.ImageOps
import numpy as np
import requests
import torch
from PIL import Image
from typing import Callable, Union


def set_nested_item(dataDict, mapList, value):
    """Set item in nested dictionary"""
    """
    Example: the mapList contains the name of each key ['injection','self-attn']
            this method will change the content in dataDict['injection']['self-attn'] with value

    """
    for k in mapList[:-1]:
        dataDict = dataDict[k]
    dataDict[mapList[-1]] = value


def merge_sweep_config(base_config, update):
    """Merge the updated parameters into the base config"""

    if base_config is None:
        raise ValueError("Base config is None")
    if update is None:
        raise ValueError("Update config is None")
    for key in update.keys():
        map_list = key.split("--")
        set_nested_item(base_config, map_list, update[key])
    return base_config


# Adapt from https://github.com/castorini/daam
def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if
                         tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise Exception(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


def extract_data(input_string: str) -> list:
    print("input_string:", input_string)
    """
    Extract data from a string pattern where contents in () are separated by ;
    The first item in each () is considered as 'ref' and the rest as 'gen'.

    Args:
    - input_string (str): The input string pattern.

    Returns:
    - list: A list of dictionaries containing 'ref' and 'gen'.
    """
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, input_string)

    data = []
    for match in matches:
        parts = [x.strip() for x in match.split(';')]
        ref = parts[0].strip()
        gen = parts[1].strip()
        data.append({'ref': ref, 'gen': gen})

    return data


def generate_hash_key(image, prompt=""):
    """
    Generate a hash key for the given image and prompt.
    """
    byte_array = io.BytesIO()
    image.save(byte_array, format='JPEG')

    # Get byte data
    image_byte_data = byte_array.getvalue()

    # Combine image byte data and prompt byte data
    combined_data = image_byte_data + prompt.encode('utf-8')

    sha256 = hashlib.sha256()
    sha256.update(combined_data)
    return sha256.hexdigest()


def save_data(data, folder_path, key):
    """
    Save data to a file, using key as the file name
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{key}.pt")

    torch.save(data, file_path)


def get_data(folder_path, key):
    """
    Get data from a file, using key as the file name
    :param folder_path:
    :param key:
    :return:
    """

    file_path = os.path.join(folder_path, f"{key}.pt")
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        return None


def PILtoTensor(data: Image.Image) -> torch.Tensor:
    return torch.tensor(np.array(data)).permute(2, 0, 1).unsqueeze(0).float()


def TensorToPIL(data: torch.Tensor) -> Image.Image:
    return Image.fromarray(data.squeeze().permute(1, 2, 0).numpy().astype(np.uint8))


# Adapt from https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/utils/loading_utils.py#L9
def load_image(
        image: Union[str, PIL.Image.Image], convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


def load_ckpt_pca_list(config_path):
    """
    Load the checkpoint and pca basis list from the config file
    :return:
    models : Dict: The dictionary of the model checkpoints
    pca_basis_dict : List : The list of the pca basis
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    # load from config
    with open(config_path, 'r') as f:
        gradio_config = yaml.safe_load(f)

    models: Dict = gradio_config['checkpoints']
    pca_basis_dict: Dict = dict()
    # remove non-exist model
    for model_version in list(models.keys()):
        for model_name in list(models[model_version].keys()):
            if "naive" not in model_name and not os.path.isfile(models[model_version][model_name]["path"]):
                models[model_version].pop(model_name)
            else:
                # Add the path of PCA basis to the pca_basis dict
                basis_dict = models[model_version][model_name]["pca_basis"]
                for key in list(basis_dict.keys()):
                    if not os.path.isfile(basis_dict[key]):
                        basis_dict.pop(key)
                if model_version not in pca_basis_dict.keys():
                    pca_basis_dict[model_version]: Dict = dict()
                if model_name not in pca_basis_dict[model_version].keys():
                    pca_basis_dict[model_version][model_name]: Dict = dict()
                pca_basis_dict[model_version][model_name].update(basis_dict)

    return models, pca_basis_dict
