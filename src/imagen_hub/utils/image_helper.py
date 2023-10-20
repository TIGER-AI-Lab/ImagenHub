from typing import Union, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import os
import requests



def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if (size != None):
        image = image.resize(size, Image.LANCZOS)
    return image


def check_format(source_path: Union[str, os.PathLike]) -> str:
    """
    Check the file format of the given source path.

    Args:
        source_path (Union[str, os.PathLike]): Path of the file to check its format.

    Returns:
        str: 'folder' if directory, 'image' if image, 'video' if video, and None if unrecognized format.
    """
    if (os.path.isdir(source_path)):
        return 'folder'
    # include image suffixes
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'
    # include video suffixes
    VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'
    if (source_path.split(".")[-1] in IMG_FORMATS):
        return 'image'
    if (source_path.split(".")[-1] in VID_FORMATS):
        return 'video'

    return None


def check_is_image(source_path: Union[str, os.PathLike]) -> bool:
    """
    Check if the given source path points to an image.

    Args:
        source_path (Union[str, os.PathLike]): Path of the file to check.

    Returns:
        bool: True if it's an image, False otherwise.
    """
    if (not os.path.isdir(source_path)):
        IMG_FORMATS = 'jpeg', 'jpg', 'png'  # include image suffixes
        if (source_path.split(".")[-1] in IMG_FORMATS):
            return True
    return False




def load_512(image_path: Union[str, os.PathLike], left: int = 0, right: int = 0, top: int = 0, bottom: int = 0, return_type: str = "numpy"):
    """
    Load, resize, and center-crop an image to a 512x512 resolution.
    Copied from Null Text Inversions' load image function
    modified from https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb

    Args:
        image_path (Union[str, os.PathLike]): Path to the image or a NumPy array.
        left (int, optional): Number of pixels to crop from the left. Defaults to 0.
        right (int, optional): Number of pixels to crop from the right. Defaults to 0.
        top (int, optional): Number of pixels to crop from the top. Defaults to 0.
        bottom (int, optional): Number of pixels to crop from the bottom. Defaults to 0.
        return_type (str, optional): Desired return format - "numpy" for a numpy array or "PIL" for a PIL Image. Defaults to "numpy".

    Returns:
        Union[np.ndarray, Image.Image]: Resized and cropped image in the specified format.
    """
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = Image.fromarray(image).resize((512, 512))
    if return_type == "numpy":
        image = np.array(image)
    return image


# Convert RGBA image to 0-1 mask, used in mask-guided image editing models.
def rgba_to_01_mask(image_rgba: Image.Image, reverse: bool = False, return_type: str = "numpy"):
    """
    Convert an RGBA image to a 0-1 binary mask.

    Args:
        image_rgba (Image.Image): The input RGBA image.
        reverse (bool, optional): If True, reverse the binary mask values. Defaults to False.
        return_type (str, optional): Format for the returned mask - "numpy" or "PIL". Defaults to "numpy".

    Returns:
        Union[np.ndarray, Image.Image]: Binary mask in the specified format.
    """
    alpha_channel = np.array(image_rgba)[:, :, 3]
    image_bw = (alpha_channel != 255).astype(np.uint8)
    if reverse:
        image_bw = 1 - image_bw
    mask = image_bw[np.newaxis, :, :]
    if return_type == "numpy":
        return mask
    else: # return PIL image
        mat = np.reshape(mask,(mask.shape[1],mask.shape[2]))
        mask = Image.fromarray(np.uint8(mat * 255) , 'L').convert('RGB')
        return mask


def highlight_masked_region(image_gen: Image.Image, mask: np.ndarray, dark_factor: float = 0.8, light_factor: float = 1.1, reverse: bool = False):
    """
    Highlight the masked region in the generated image by brightening the masked area and darkening the rest.

    Args:
        image_gen (Image.Image): The input image to highlight. Image in PIL format.
        mask (np.ndarray): A binary mask as a NumPy array with values in the range[0, 255], where masked areas have values = 0 and unmasked areas have values > 0.
        dark_factor (float, optional): Adjustment factor for non-masked regions. Defaults to 0.8. Higher values denotes higher brightness.
        light_factor (float, optional): Adjustment factor for masked regions. Defaults to 1.1. Higher values denotes higher brightness.
        reverse (bool, optional): If True, reverse the mask before applying. Defaults to False.

    Returns:
        Image.Image: Image with the masked region highlighted.
    """

    image_gen = np.array(image_gen, dtype=np.float32)  # Convert to float32
    highlighted_image = image_gen.copy()
    if reverse:
        # Brighten the masked region
        highlighted_image[mask == 0] *= dark_factor
        # Darken the unmasked region
        highlighted_image[mask > 0] *= light_factor

    else:
        highlighted_image[mask == 0] *= light_factor
        highlighted_image[mask > 0] *= dark_factor

    # Ensure pixel values are within [0, 255]
    highlighted_image = np.clip(highlighted_image, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    highlighted_image_pil = Image.fromarray(highlighted_image)

    return highlighted_image_pil
