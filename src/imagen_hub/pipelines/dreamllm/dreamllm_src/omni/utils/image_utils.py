import io
import os
from typing import Literal, TypeAlias

import megfile
import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
import torch
from typing import Union
from .loguru import logger

"""
- pil: `PIL.Image.Image`, size (w, h), seamless conversion between `uint8`
- np: `np.ndarray`, shape (h, w, c), default `np.uint8`
- pt: `torch.Tensor`, shape (c, h, w), default `torch.uint8`
"""
ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
ImageTypeStr: TypeAlias = Literal["pil", "np", "pt"]
ImageFormat: TypeAlias = Literal["JPEG", "PNG"]
DataFormat: TypeAlias = Literal["255", "01", "11"]


def check_image_type(image: ImageType):
    if not (isinstance(image, PIL.Image.Image) or isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)):
        raise TypeError(f"`image` should be PIL Image, ndarray or Tensor. Got `{type(image)}`.")


def load_image(
    image: Union[str, os.PathLike]| PIL.Image.Image,
    *,
    output_type: ImageTypeStr = "pil",
) -> ImageType:
    """
    Loads `image` to a PIL Image, NumPy array or PyTorch tensor.

    Args:
        image (str | PIL.Image.Image): The path to image or PIL Image.
        mode (ImageMode, optional): The mode to convert to. Defaults to None (no conversion).
            The current version supports all possible conversions between "L", "RGB", "RGBA".
        output_type (ImageTypeStr, optional): The type of the output image. Defaults to "pil".
            The current version supports "pil", "np", "pt".

    Returns:
        ImageType: The loaded image in the given type.
    """
    timeout = 10
    # Load the `image` into a PIL Image.
    if isinstance(image, str) or isinstance(image, os.PathLike):
        if image.startswith("http://") or image.startswith("https://"):
            try:
                image = PIL.Image.open(requests.get(image, stream=True, timeout=timeout).raw)
            except requests.exceptions.Timeout:
                raise ValueError(f"HTTP request timed out after {timeout} seconds")
        elif image.startswith("s3"):
            with megfile.smart_open(image, "rb") as f:
                bytes_data = f.read()
            image = PIL.Image.open(io.BytesIO(bytes_data), "r")
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://`, `https://` or `s3+[profile]://`, and `{image}` is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(f"`image` must be a path or PIL Image, got `{type(image)}`")

    # Automatically adjust the orientation of the image to match the direction it was taken.
    image = PIL.ImageOps.exif_transpose(image)

    # support_mode = ["L", "RGB", "RGBA", "CMYK", "P", "I", "LA"]
    # if image.mode not in support_mode:
    #     raise ValueError(f"Only support mode in `{support_mode}`, got `{image.mode}`")

    # add white background for RGBA images, and convert to RGB
    if image.mode == "RGBA":
        background = PIL.Image.new("RGBA", image.size, "white")
        image = PIL.Image.alpha_composite(background, image).convert("RGB")

    image = image.convert("RGB")

    if output_type == "pil":
        image = image
    elif output_type == "np":
        image = to_np(image)
    elif output_type == "pt":
        image = to_pt(image)
    else:
        raise ValueError(f"`output_type` must be one of `{ImageTypeStr}`, got `{output_type}`")

    return image


def save_image(
    image: ImageType,
    *,
    path: Union[str, os.PathLike],
    format: Union[ImageFormat, None]= None,
    force_overwrite: bool = True,
) -> str:
    """
    Save PIL Image to path.

    Args:
        image (PIL.Image.Image): The PIL image to be saved.
        path (str): The path to save the image, support s3.
        format (str, optional): Defaults to None. If set, will save the image with the given format.

    Returns:
        str: The path to the saved image.
    """
    check_image_type(image)

    if isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
        image = to_pil(image)
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError("Incorrect format used for image. Should be a PIL image, a NumPy array, or a PyTorch tensor.")

    # splitext: split the path into a pair (root, .ext)
    root, extension = os.path.splitext(path)
    if format is not None:
        if extension[1:].lower() == "jpg":
            _extension = "jpeg"
        else:
            _extension = extension[1:].lower()
        assert _extension == format.lower(), f"extension `{extension[1:]}` does not match format `{format}`"

    if path.startswith("s3"):
        buffered = io.BytesIO()
        image.save(buffered, format=format)

        if megfile.s3_isfile(path):
            if force_overwrite:
                logger.info(f"Overwriting existing file: {path}")
            else:
                count = 0
                valid_path = path
                while megfile.s3_isfile(valid_path):
                    valid_path = root + "_" + str(count) + extension
                    count += 1
                logger.info(f"File already exists: {path}, saving to {valid_path}")
                path = valid_path

        with megfile.smart_open(path, "wb") as f:
            f.write(buffered.getvalue())

    else:
        path_dir = os.path.dirname(path)
        if path_dir != "":
            os.makedirs(path_dir, exist_ok=True)

        if os.path.isfile(path):
            if force_overwrite:
                logger.info(f"Overwriting existing file: {path}")
            else:
                count = 0
                valid_path = path
                while os.path.isfile(valid_path):
                    valid_path = root + "_" + str(count) + extension
                    count += 1
                logger.info(f"File already exists: {path}, saving to {valid_path}")
                path = valid_path

        image.save(path, format=format)

    return path


def to_pil(image: ImageType, image_mode: Union[DataFormat, None]= None) -> PIL.Image.Image:
    """
    Convert a NumPy array or a PyTorch tensor to a PIL image.
    """
    check_image_type(image)

    if isinstance(image, PIL.Image.Image):
        return image

    elif isinstance(image, np.ndarray):
        image = normalize_np(image, image_mode)

    elif isinstance(image, torch.Tensor):
        image = normalize_pt(image, image_mode)

        image = image.cpu().permute(1, 2, 0).numpy()
        assert image.dtype == np.uint8, f"Supposed to convert `torch.uint8` to `np.uint8`, but got `{image.dtype}`"

    mode_map = {1: "L", 3: "RGB"}
    mode = mode_map[image.shape[-1]]

    if image.shape[-1] == 1:
        image = image[:, :, 0]

    return PIL.Image.fromarray(image, mode=mode)


def to_np(image: ImageType, image_mode: Union[DataFormat, None]= None) -> np.ndarray:
    """
    Convert a PIL image or a PyTorch tensor to a NumPy array.
    """
    check_image_type(image)

    if isinstance(image, PIL.Image.Image):
        image = np.array(image, np.uint8, copy=True)

    if isinstance(image, np.ndarray):
        image = normalize_np(image, image_mode)

    elif isinstance(image, torch.Tensor):
        image = normalize_pt(image, image_mode)

        image = image.cpu().permute(1, 2, 0).numpy()
        assert image.dtype == np.uint8, f"Supposed to convert `torch.uint8` to `np.uint8`, but got `{image.dtype}`"

    return image


def to_pt(image: ImageType, image_mode: Union[DataFormat, None]= None) -> torch.Tensor:
    """
    Convert a PIL image or a NumPy array to a PyTorch tensor.
    """
    check_image_type(image)

    if isinstance(torch.Tensor):
        image = normalize_pt(image, image_mode)
        return image

    # convert PIL Image to NumPy array
    if isinstance(image, PIL.Image.Image):
        image = np.array(image, np.uint8, copy=True)

    image = normalize_np(image, image_mode)

    image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
    assert image.dtype == torch.uint8, f"Supposed to convert `np.uint8` to `torch.uint8`, but got `{image.dtype}`"
    return image


def normalize_np(image: np.ndarray, image_mode: Union[DataFormat, None]= None) -> np.ndarray:
    """
    Normalize a NumPy array to the standard format of shape (h, w, c) and uint8.
    """
    if image.ndim not in {2, 3}:
        raise ValueError(f"`image` should be 2 or 3 dimensions. Got {image.ndim} dimensions.")

    elif image.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        image = np.expand_dims(image, 2)

    if image.shape[-1] not in {1, 3}:
        raise ValueError(f"`image` should have 1 (`L`) or 3 (`RGB`) channels. Got {image.shape[-1]} channels.")

    image = to_dataformat(image, image_mode=image_mode, mode="255")

    return image


def normalize_pt(image: torch.Tensor, image_mode: Union[DataFormat, None]= None) -> torch.Tensor:
    """
    Normalize a PyTorch tensor to the standard format of shape (c, h, w) and uint8.
    """
    if image.ndimension() not in {2, 3}:
        raise ValueError(f"`image` should be 2 or 3 dimensions. Got {image.ndimension()} dimensions.")

    elif image.ndimension() == 2:
        # if 2D image, add channel dimension (CHW)
        image = image.unsqueeze(0)

    # check number of channels
    if image.shape[-3] not in {1, 3}:
        raise ValueError(f"`image` should have 1 (`L`) or 3 (`RGB`) channels. Got {image.shape[-3]} channels.")

    image = to_dataformat(image, image_mode=image_mode, mode="255")

    return image


def to_dataformat(
    image: ImageType,
    *,
    image_mode: Union[DataFormat, None]= None,
    mode: DataFormat = "255",
) -> Union[np.ndarray, torch.Tensor]:
    check_image_type(image)

    # convert PIL Image to NumPy array
    if isinstance(image, PIL.Image.Image):
        image = np.array(image, np.uint8, copy=True)
        image_mode = "255"

    # guess image mode
    if image.dtype == np.uint8 or image.dtype == torch.uint8:
        guess_image_mode = "255"
    elif image.dtype == np.float32 or image.dtype == np.float16 or image.dtype == torch.float32 or image.dtype == torch.float16:
        if image.min() < 0.0:
            guess_image_mode = "11"
        else:
            guess_image_mode = "01"
    else:
        raise ValueError(f"Unsupported dtype `{image.dtype}`")

    if image_mode is None:
        image_mode = guess_image_mode
    else:
        if guess_image_mode != image_mode:
            logger.warning(f"Guess image mode is `{guess_image_mode}`, but image mode is `{image_mode}`")

    if isinstance(image, np.ndarray):
        if image_mode == "255" and mode != "255":
            image = (image.astype(np.float32) / 255).clamp(0, 1)
            if mode == "11":
                image = (image * 2 - 1).clamp(-1, 1)

        elif image_mode == "01" and mode != "01":
            if mode == "255":
                image = image.clamp(0, 1)
                image = (image * 255).round().astype(np.uint8)
            elif mode == "11":
                image = (image * 2 - 1).clamp(-1, 1)

        elif image_mode == "11" and mode != "11":
            image = (image / 2 + 0.5).clamp(0, 1)
            if mode == "255":
                image = (image * 255).round().astype(np.uint8)

    elif isinstance(image, torch.Tensor):
        if image_mode == "255" and mode != "255":
            image = image.to(dtype=torch.float32).div(255).clamp(0, 1)
            if mode == "11":
                image = (image * 2 - 1).clamp(-1, 1)

        elif image_mode == "01" and mode != "01":
            if mode == "255":
                image = image.clamp(0, 1)
                image = (image * 255).round().to(dtype=torch.uint8)
            elif mode == "11":
                image = (image * 2 - 1).clamp(-1, 1)

        elif image_mode == "11" and mode != "11":
            image = (image / 2 + 0.5).clamp(0, 1)
            if mode == "255":
                image = image.mul(255).round().to(dtype=torch.uint8)

    return image


def images2grid(
    images: Union[ImageType, list[ImageType]],
    *,
    num_rows: int = 1,
    offset: Union[float, None]= None,
):
    if not isinstance(images, list):
        images = [images]

    if not isinstance(images[0], np.ndarray):
        images = [to_np(image) for image in images]

    h, w, _ = images[0].shape
    if offset is None:
        offset = int(h * 0.02)

    num_images = len(images)

    num_empty = num_images % num_rows
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255

    images = images + [empty_images] * num_empty

    num_cols = num_images // num_rows
    grid_image = (
        np.ones((h * num_rows + offset * (num_rows - 1), w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    )

    for i in range(num_rows):
        for j in range(num_cols):
            grid_image[i * (h + offset) : i * (h + offset) + h :, j * (w + offset) : j * (w + offset) + w] = images[
                i * num_cols + j
            ]

    return grid_image


if __name__ == "__main__":
    tensor_uint8 = torch.rand(2, 3).to(torch.uint8)
    tensor_uint8.to(torch.float32).div(255)
    paths = ["./l-test.png", "./rgb-test.jpg", "./rgba-test.png"]
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
        "https://picx.zhimg.com/b8eee1e044dc053c4ac5f09c6e0a3886_r.jpg?source=1940ef5c",
        "https://img-blog.csdnimg.cn/20201104220504131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pvc2VwaF9fTGFncmFuZ2U=,size_16,color_FFFFFF,t_70#pic_center",
    ]
    mode = ["L", "RGB", "RGB"]

    for _mode, path in zip(mode, paths):
        image = load_image(path)
        image_path = save_image(image, path="./test/test_path.jpg", format="JPEG", force_overwrite=False)
        image_save = load_image(image_path)
        assert image.mode == _mode, f"image.mode: {image.mode}, _mode: {_mode}"
        assert image.mode == image_save.mode, f"image.mode: {image.mode}, image_save.mode: {image_save.mode}"

    for _mode, url in zip(mode, urls):
        image = load_image(url)
        image_path = save_image(image, path="./test/test_url.jpg", format="JPEG", force_overwrite=False)
        image_save = load_image(image_path)
        assert image.mode == _mode, f"image.mode: {image.mode}, _mode: {_mode}"
        assert image.mode == image_save.mode, f"image.mode: {image.mode}, image_save.mode: {image_save.mode}"

    # load, save, NumPy and PyTorch
    for _mode, path in zip(mode, paths):
        image = load_image(path, mode="L")
        assert image.mode == "L", f"image.mode: {image.mode}, _mode: {_mode}"

        image = load_image(path, mode="RGB")
        assert image.mode == "RGB", f"image.mode: {image.mode}, _mode: {_mode}"

        image = load_image(path, output_type="np")
        if _mode == "L":
            assert image.shape[-1] == 1 and image.ndim == 3, f"image.shape: {image.shape}, _mode: {_mode}"
        else:
            assert image.shape[-1] == 3 and image.ndim == 3, f"image.shape: {image.shape}, _mode: {_mode}"
        assert image.dtype == np.uint8, f"image.dtype: {image.dtype}, _mode: {_mode}"
        assert (
            image.min() >= 0 and image.max() <= 255
        ), f"image.min(): {image.min()}, image.max(): {image.max()}, _mode: {_mode}"
        save_image(image, path="./test/test_save_np.jpg", format="JPEG", force_overwrite=False)

        image = load_image(path, output_type="pt")
        if _mode == "L":
            assert image.shape[-3] == 1 and image.ndimension() == 3, f"image.shape: {image.shape}, _mode: {_mode}"
        else:
            assert image.shape[-3] == 3 and image.ndimension() == 3, f"image.shape: {image.shape}, _mode: {_mode}"
        assert image.dtype == torch.uint8, f"image.dtype: {image.dtype}, _mode: {_mode}"
        assert (
            image.min() >= 0 and image.max() <= 255
        ), f"image.min(): {image.min()}, image.max(): {image.max()}, _mode: {_mode}"
        save_image(image, path="./test/test_save_pt.jpg", format="JPEG", force_overwrite=False)
