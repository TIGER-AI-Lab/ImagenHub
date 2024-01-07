from imagen_hub.loader.benchmark_loader import dump_dataset_info, load_control_guided_ig_dataset
from imagen_hub.utils.save_image_helper import save_pil_image, get_concat_pil_images
from typing import Union, Optional, Callable
import os
from PIL import Image
from tqdm import tqdm

def infer_control_guided_ig_bench(model,
                             result_folder: str = 'results',
                             experiment_name: str = "Exp_Control-Guided_IG",
                             overwrite_model_outputs: bool = False,
                             overwrite_inputs: bool = False,
                             limit_images_amount: Optional[int] = None):
    """
    Performs inference on ImagenHub dataset using a control-guided image generation model.

    This function infers images based on the control type and associated text for each sample.
    The results, including the guide images and the inferred images, are saved in the specified directories.

    Args:
        model: Model instance for control-guided image generation.
               Expected to have a method 'infer_one_image' for inferencing.
        result_folder (str, optional): Root directory where the results will be saved.
               Defaults to 'results'.
        experiment_name (str, optional): Name of the sub-directory inside 'result_folder'
               where the results for this experiment will be saved. Defaults to "Exp_Control-Guided_IG".
        overwrite_model_outputs (bool, optional): If True, pre-existing model outputs will
               be overwritten. Useful for resuming interrupted runs. Defaults to False.
        overwrite_inputs (bool, optional): If True, will overwrite any pre-existing input
               samples. Typically set to False unless there's a reason to update the inputs. Defaults to False.
        limit_images_amount (int, optional): Specifies the maximum number of images to process
               from the dataset. If None, all images will be processed.

    Returns:
        None. The results, including guide images and inferred images, are saved in the designated directories.

    Notes:
        The function reads samples from the dataset, uses the provided model to infer images based
        on guide images and associated text prompts, and then saves the results in the designated directories.
    """
    dataset, dataset_name = load_control_guided_ig_dataset(with_name_att=True)
    data = dataset['eval']
    del dataset

    def process_dataset_uid(sample):
        imd_id = sample['img_id']
        control_type = sample["control_type"]
        return f"sample_{imd_id}_control_{control_type}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(data,
                          required_attributes=['text',
                                               'control_type'],
                          uid_preprocess_fn=process_dataset_uid,
                          save_folder_path=os.path.join(
                              result_folder, experiment_name),
                          filename_wo_ext='dataset_lookup')

    print("========> Running Benchmark Dataset:", experiment_name, "| Model:", model.__class__.__name__)
    index = 0
    for sample in tqdm(data):
        file_basename = process_dataset_uid(sample)
        gt_folder = os.path.join(result_folder, experiment_name, "GroundTruth")
        input_folder = os.path.join(result_folder, experiment_name, "input")
        dest_folder = os.path.join(result_folder, experiment_name, model.__class__.__name__)
        dest_file = os.path.join(dest_folder, file_basename)
        if overwrite_model_outputs or not os.path.exists(dest_file):
            sample_input = sample['guide'].resize((512, 512), Image.LANCZOS)
            instruction = sample['text']
            output = model.infer_one_image(src_image=sample_input,
                                        prompt=instruction)
            output = output.resize((512, 512), Image.LANCZOS)
            sample_gt = sample['image'].resize((512, 512), Image.LANCZOS)
            save_pil_image(sample_input, input_folder, file_basename, overwrite=overwrite_inputs)
            save_pil_image(sample_gt, gt_folder, file_basename, overwrite=overwrite_inputs)
            save_pil_image(output, dest_folder, file_basename)
        else:
            print("========> Skipping", dest_file, ", it already exists")
        index += 1
        if limit_images_amount is not None and (index >= limit_images_amount):
            break