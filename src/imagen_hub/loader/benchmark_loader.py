from datasets import load_dataset
from typing import List, Union, Callable
from tqdm import tqdm
import json
import os
import csv


def dump_dataset_info(your_dataset,
                      required_attributes: List,
                      uid_preprocess_fn: Callable,
                      save_folder_path: Union[str, os.PathLike],
                      filename_wo_ext: Union[str, os.PathLike]="dataset_lookup",
                      verbose=True):
    """
    Dump dataset text info according to specified attributes.

    Args:
        your_dataset : The dataset containing the data.
        required_attributes (List): List of attribute names that you want to keep track of.
        uid_preprocess_fn (Callable): A function that takes a `sample` (a single row of the dataset) as a parameter.
            This function should return the uid, expected to be the image filename.
        save_folder_path (Union[str, os.PathLike]): Path to the folder where the json dump will be saved.
        filename_wo_ext (Union[str, os.PathLike]): Filename for the saved json file without its extension.

    Example::
    
        def process_uid(sample):
            imd_id = sample['img_id']
            turn_index = sample['turn_index']
            return f"sample_{imd_id}_{turn_index}.jpg"
        my_dataset, dataset_name = load_data()
        data = my_dataset['dev']
        dump_dataset_info(data,
                            required_attributes=['instruction',
                                                 'source_global_caption',
                                                 'target_global_caption'],
                            uid_preprocess_fn=process_uid,
                            save_folder_path=os.path.join('results', dataset_name),
                            filename_wo_ext='dataset_lookup')
    """
    if (verbose):
        print(f"=======> Dumping dataset text info to json file")
    data = your_dataset
    complete_dict = dict()
    uid_list = []

    for sample in tqdm(data):
        row_dict = dict()
        uid = uid_preprocess_fn(sample)
        for attribute in required_attributes:
            row_dict[str(attribute)] = sample[attribute]
        complete_dict[uid] = row_dict
        uid_list.append([uid])

    # convert into JSON:
    json_obj = json.dumps(complete_dict, indent=4)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    # Writing to save_json_path json file
    saving_path = os.path.join(save_folder_path, f'{filename_wo_ext}.json')
    with open(saving_path, "w") as outfile:
        outfile.write(json_obj)

    # Writing to a csv file with all the uid s
    with open(os.path.join(save_folder_path, f'{filename_wo_ext}.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(['uid'])
        # write multiple rows data
        writer.writerows(uid_list)

    if (verbose):
        print(f"=======> Saved info to {save_folder_path}")


# Text-To-Image Generation Benchmark

def load_text_guided_ig_dataset(with_name_att: bool = False, name_att: str = "ImagenHub_Text-to-Image_Bench"):
    """
    Load the Text-to-Image Generation Benchmark dataset from HuggingFace.

    Args:
        with_name_att (bool, optional): If True, returns the name attribute. Defaults to False.
        name_att (str, optional): Name of the dataset attribute. Defaults to "ImagenHub_Text-to-Image_Bench".

    Returns:
        tuple: Dataset and optionally the name attribute.

    Dataset Splits:
        eval: 197
        DrawBench_trimmed: 77
        DiffusionDB_trimmed: 40
        Realism: 40
        ABC_trimmed: 40
    """
    data = load_dataset("ImagenHub/Text_to_Image")
    return data, name_att if with_name_att else data


# Text-Guided Image Editing Benchmark

def load_text_guided_ie_dataset(with_name_att: bool = False, name_att: str = "ImagenHub_Text-Guided-Image-Editing_Bench"):
    """
    Load the Text-Guided Image Editing Benchmark dataset from HuggingFace.

    Args:
        with_name_att (bool, optional): If True, returns the name attribute. Defaults to False.
        name_att (str, optional): Name of the dataset attribute.

    Returns:
        tuple: Dataset and optionally the name attribute.

    Dataset Splits:
        dev: 528
        filtered: 179
        extra: 249
    """
    data = load_dataset("ImagenHub/Text_Guided_Image_Editing")
    return data, name_att if with_name_att else data

# Mask-Guided Image Editing Benchmark


def load_mask_guided_ie_dataset(with_name_att: bool = False, name_att: str = "ImagenHub_Mask-Guided-Image-Editing_Bench"):
    """
    Load the Mask-Guided Image Editing Benchmark dataset from HuggingFace.

    Args:
        with_name_att (bool, optional): If True, returns the name attribute. Defaults to False.
        name_att (str, optional): Name of the dataset attribute.

    Returns:
        tuple: Dataset and optionally the name attribute.

    Dataset Splits:
        dev: 528
        filtered: 179
        extra: 249
    """
    data = load_dataset("ImagenHub/Mask_Guided_Image_Editing")
    return data, name_att if with_name_att else data

# Control-Guided Image Generation Benchmark


def load_control_guided_ig_dataset(with_name_att=False, name_att="ImagenHub_Control-Guided-Image-Generation_Bench"):
    """
    Load the Control-Guided Image Generation Benchmark dataset.

    Args:
        with_name_att (bool, optional): If True, returns the name attribute. Defaults to False.
        name_att (str, optional): Name of the dataset attribute.

    Returns:
        tuple: Dataset and optionally the name attribute.

    Dataset Splits:
        full: 5 condition type * 100 samples = 500 rows data
        eval: 5 condition type * 30 samples = 150 rows data
        extra: 5 condition type * 70 samples = 350 rows data
    """
    data = load_dataset("ImagenHub/Control_Guided_Image_Generation")
    return data, name_att if with_name_att else data

# Multi-Subject-Driven Image Generation Benchmark


def load_multi_concept_ic_dataset(with_name_att: bool = False, name_att: str = "ImagenHub_Multi-Subject-Driven-Image-Generation_Bench"):
    """
    Load the Multi-Subject-Driven Image Generation Benchmark dataset.
    Args:
        with_name_att (bool): If True, returns dataset with its name.
        name_att (str): Dataset name.

    Returns:
        Tuple: Dataset and its name (if `with_name_att` is True), otherwise just the dataset.

    Dataset Splits:
        train: 50 rows
        test: (should be eval), 102 rows
    """
    data = load_dataset("ImagenHub/Multi-Subject-Concepts")
    data['test'] = load_dataset(
        'ImagenHub/Multi_Subject_Driven_Image_Generation')['train']
    return data, name_att if with_name_att else data

# Subject-Driven Image Generation Benchmark


def load_subject_driven_ig_dataset(with_name_att: bool = False, name_att: str = "ImagenHub_Subject-Driven-Image-Generation_Bench"):
    """
    Load the Subject-Driven Image Generation Benchmark dataset.

    Args:
        with_name_att (bool): If True, returns dataset with its name.
        name_att (str): Dataset name.

    Returns:
        Tuple: Dataset and its name (if `with_name_att` is True), otherwise just the dataset.

    Dataset Splits:
        train: 158 rows for 30 subjects
        eval: 150 rows = 30 subjects * 5 prompts
    """
    data = load_dataset("ImagenHub/DreamBooth-Concepts")
    data['eval'] = load_dataset(
        'ImagenHub/Subject_Driven_Image_Generation')['eval']
    return data, name_att if with_name_att else data

# Subject-Driven Image Editing Benchmark


def load_subject_driven_ie_dataset(with_name_att: bool = False, name_att: str = "ImagenHub_Subject-Driven-Image-Editing_Bench"):
    """
    Load the Subject-Driven Image Editing Benchmark dataset.

    Args:
        with_name_att (bool): If True, returns dataset with its name.
        name_att (str): Dataset name.

    Returns:
        Tuple: Dataset and its name (if `with_name_att` is True), otherwise just the dataset.

    Dataset Splits:
        train: 158 rows for 30 subjects
        eval: 154 = 7 samples * 22 objects
    """
    data = load_dataset("ImagenHub/DreamBooth-Concepts")
    data['eval'] = load_dataset(
        'ImagenHub/Subject_Driven_Image_Editing')['eval']
    return data, name_att if with_name_att else data
