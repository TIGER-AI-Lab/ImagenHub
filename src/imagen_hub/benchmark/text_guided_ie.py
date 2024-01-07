from imagen_hub.loader.benchmark_loader import dump_dataset_info, load_text_guided_ie_dataset
from imagen_hub.utils.save_image_helper import save_pil_image, get_concat_pil_images
from typing import Union, Optional, Callable
import os
from PIL import Image
from tqdm import tqdm

def infer_text_guided_ie_bench(model,
                        result_folder: str = 'results',
                        experiment_name: str = "Exp_Text-Guided_IE",
                        overwrite_model_outputs: bool = False,
                        overwrite_inputs: bool = False,
                        limit_images_amount: Optional[int] = None):
    """
    Performs inference on the ImagenHub dataset using the provided text-guided image editing model.

    Args:
        model: Instance of the Text-Guided Image Editing model or the Mask-Guided Image Editing model.
            Expected to have a method 'infer_one_image' for inferencing.
        result_folder (str, optional): Path to the root directory where the results should be saved.
            Defaults to 'results'.
        experiment_name (str, optional): Name of the folder inside 'result_folder' where results for
            this particular experiment will be stored. Defaults to "Exp_Text-Guided_IE".
        overwrite_model_outputs (bool, optional): If set to True, will overwrite any pre-existing
            model outputs. Useful for resuming runs. Defaults to False.
        overwrite_inputs (bool, optional): If set to True, will overwrite any pre-existing input
            samples. Typically, should be set to False unless there's a need to update the inputs.
            Defaults to False.
        limit_images_amount (int, optional): Limits the number of images to be processed. If set to
            None, all images in the dataset will be processed.

    Returns:
        None. Results are saved in the specified directory.

    Notes:
        The function processes each sample from the dataset, generates a unique ID for it, and saves
        the results in the respective directories. The results include the processed image and any
        associated metadata.
    """
    dataset, dataset_name = load_text_guided_ie_dataset(with_name_att=True)
    data = dataset['filtered']
    del dataset

    def process_dataset_uid(sample):
        imd_id = sample['img_id']
        turn_index = sample['turn_index']
        return f"sample_{imd_id}_{turn_index}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(data,
                        required_attributes=['source_global_caption',
                                            'instruction',
                                            'target_global_caption'],
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
            print("========> Inferencing", dest_file)
            sample_input = sample['source_img'].resize(
                (512, 512), Image.LANCZOS)
            sample_gt = sample['target_img'].resize(
                (512, 512), Image.LANCZOS)
            instruction = sample['instruction']
            src_caption = sample['source_global_caption']
            tgt_caption = sample['target_global_caption']
            output = model.infer_one_image(src_image=sample_input,
                                            src_prompt=src_caption,
                                            target_prompt=tgt_caption,
                                            instruct_prompt=instruction)

            output = output.resize((512, 512), Image.LANCZOS)
            save_pil_image(sample_input, input_folder, file_basename, overwrite=overwrite_inputs)
            save_pil_image(sample_gt, gt_folder, file_basename, overwrite=overwrite_inputs)
            save_pil_image(output, dest_folder, file_basename)
        else:
            print("========> Skipping", dest_file, ", it already exists")
        index += 1
        if limit_images_amount is not None and (index >= limit_images_amount):
            break