import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import os
from PIL import Image
from typing import Union


def save_tensor_image(tensor_image: torch.Tensor, dest_folder: Union[str, os.PathLike], filename: Union[str, os.PathLike], filetype: str = "jpg"):
    """
    Save a tensor image to the specified destination folder with a given filename and filetype.

    Args:
        tensor_image (torch.Tensor): The image tensor of shape [B, 3, H, W].
        dest_folder (Union[str, os.PathLike]): Path to the destination folder.
        filename (Union[str, os.PathLike]): The desired filename.
        filetype (str, optional): The desired file extension/type (e.g., "jpg", "png"). Defaults to "jpg".

    Returns:
        None
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    save_image(tensor_image, os.path.join(
        dest_folder, f"{filename}.{filetype}"))


def save_pil_image(pil_image: Image.Image, dest_folder: Union[str, os.PathLike], filename: Union[str, os.PathLike], filetype: str = None, overwrite: bool = True):
    """
    Save a PIL Image to the specified destination folder with the given filename and filetype.

    Args:
        pil_image (PIL Image): The input PIL Image.
        dest_folder (str): Destination folder path.
        filename (str): Filename (with or without extension).
        filetype (str): File type (e.g., "jpg", "png"). Default is None. Use it when you dont have extension for filename.
        overwrite (bool): Whether to overwrite the file if it already exists.

    Returns:
        None
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if filetype is not None:
        file_path = os.path.join(dest_folder, f"{filename}.{filetype}")
    else:
        file_path = os.path.join(dest_folder, f"{filename}")

    if overwrite or not os.path.exists(file_path):
        pil_image = pil_image.convert('RGB')
        pil_image.save(file_path)


def get_mask_pil_image(mask_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a mask tensor to a PIL Image.

    Args:
        mask_tensor (torch.Tensor): Input mask tensor of shape [H, W].

    Returns:
        Image.Image: Converted mask in PIL Image format.
    """
    mask = np.array(mask_tensor).astype('uint8')
    mask = np.squeeze(mask)
    mask_img = Image.fromarray(mask * 255)
    return mask_img


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a tensor.

    Args:
        pil_img (Image.Image): Input PIL Image.

    Returns:
        torch.Tensor: Image tensor of shape [1, C, H, W].
    """
    tensor = transforms.ToTensor()(pil_img).unsqueeze_(0)
    return tensor


def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to a PIL Image.

    Args:
        tensor_img (torch.Tensor): Image tensor of shape [1, C, H, W].

    Returns:
        Image.Image: Converted image in PIL Image format.
    """
    pil = transforms.ToPILImage()(tensor_img.squeeze_(0))
    return pil


def get_concat_pil_images(images: list, direction: str = 'h') -> Image.Image:
    """
    Concatenate a list of PIL Images either horizontally or vertically.

    Args:
        images (list): List of PIL Images to be concatenated.
        direction (str, optional): Concatenation direction ('h' for horizontal, otherwise vertical). Defaults to 'h'.

    Returns:
        Image.Image: Concatenated image in PIL Image format.
    """
    if direction == 'h':
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im
    else:
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        return new_im
