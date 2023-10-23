from datasets import load_dataset
from functools import partial

def load_data(hf_ds, split, streaming=True, seed=42, buffer_size=10, get_one_data=True):
    """
    Load data from the specified Huggingface dataset and split for dummy inference.

    Args:
    - hf_ds (str): Name of the dataset in Huggingface's hub.
    - split (str): The split of the dataset to load (e.g. 'train', 'eval').
    - streaming (bool, optional): If true, datasets are streamed. Defaults to True.
    - seed (int, optional): Seed for shuffling. Defaults to 42.
    - buffer_size (int, optional): Buffer size for shuffling. Defaults to 10.
    - get_one_data (bool, optional): If true, returns the first data instance. If false, returns the whole dataset. Defaults to True.

    Returns:
    - Dataset or single data instance depending on the value of `get_one_data`.
    """
    ds = load_dataset(hf_ds, split=split, streaming=streaming).shuffle(seed=seed, buffer_size=buffer_size)
    if get_one_data:
        return next(iter(ds))
    else:
        return ds

# Partially applied functions for specific datasets and splits
load_text_guided_ig_data = partial(load_data, "ImagenHub/Text_to_Image", 'eval')
load_text_guided_ie_data = partial(load_data, "ImagenHub/Mask_Guided_Image_Editing", 'filtered')
load_mask_guided_ie_data = partial(load_data, "ImagenHub/Mask_Guided_Image_Editing", 'filtered')
load_subject_driven_ig_data = partial(load_data, "ImagenHub/Subject_Driven_Image_Generation", 'eval')
load_subject_driven_ie_data = partial(load_data, "ImagenHub/Subject_Driven_Image_Editing", 'eval')
load_control_guided_ig_data = partial(load_data, "ImagenHub/Control_Guided_Image_Generation", "eval")
load_mult_concept_ig_data = partial(load_data, "ImagenHub/Multi_Subject_Driven_Image_Generation", "train")
