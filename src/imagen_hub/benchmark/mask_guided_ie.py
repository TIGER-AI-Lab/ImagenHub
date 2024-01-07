from imagen_hub.loader.benchmark_loader import dump_dataset_info, load_mask_guided_ie_dataset
from imagen_hub.utils.save_image_helper import save_pil_image, get_concat_pil_images
from imagen_hub.utils.image_helper import rgba_to_01_mask
from typing import Union, Optional, Callable
import os
from PIL import Image
from tqdm import tqdm

def infer_mask_guided_ie_bench(model,
                        result_folder: str = 'results',
                        experiment_name: str = "Exp_Mask-Guided_IE",
                        overwrite_model_outputs: bool = False,
                        overwrite_inputs: bool = False,
                        limit_images_amount: Optional[int] = None):
    """
    Performs inference on the ImagenHub dataset using a Mask-Guided Image Editing model.

    Args:
        model: Model instance capable of text-guided or mask-guided image editing.
               Expected to have a method 'infer_one_image' for inferencing.
        result_folder (str, optional): Root directory where the results will be saved.
               Defaults to 'results'.
        experiment_name (str, optional): Name of the sub-directory inside 'result_folder'
               where the results for this particular experiment will be saved. Defaults to "Exp_Mask-Guided_IE".
        overwrite_model_outputs (bool, optional): If True, pre-existing model outputs will
               be overwritten. Useful for resuming interrupted runs. Defaults to False.
        overwrite_inputs (bool, optional): If True, will overwrite any pre-existing input
               samples. Typically set to False unless there's a reason to update the inputs. Defaults to False.
        limit_images_amount (int, optional): Specifies the maximum number of images to process
               from the dataset. If None, all images will be processed.

    Returns:
        None. The results, including input images, masks, and model outputs, are saved in the designated directory.

    Notes:
        The function reads samples from the dataset, uses the provided model to infer image edits
        based on input images, masks, and captions, and then saves the results in the designated directories.
    """
    dataset, dataset_name = load_mask_guided_ie_dataset(with_name_att=True)
    data = dataset['filtered']
    del dataset

    def process_dataset_uid(sample):
        imd_id = sample['img_id']
        turn_index = sample['turn_index']
        return f"sample_{imd_id}_{turn_index}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(data,
                        required_attributes=['instruction',
                                            'target_local_caption'],
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
            sample_mask = sample['mask_img'].resize((512, 512), Image.LANCZOS)
            tgt_caption = sample['target_local_caption']
            output = model.infer_one_image(src_image=sample_input,
                                            mask_image=sample_mask,
                                            local_mask_prompt=tgt_caption)
            save_pil_image(rgba_to_01_mask(sample_mask, return_type="PIL"), os.path.join(
                result_folder, experiment_name, "mask"), file_basename, overwrite=overwrite_inputs)

            output = output.resize((512, 512), Image.LANCZOS)
            save_pil_image(sample_input, input_folder, file_basename, overwrite=overwrite_inputs)
            save_pil_image(sample_gt, gt_folder, file_basename, overwrite=overwrite_inputs)
            save_pil_image(output, dest_folder, file_basename)
        else:
            print("========> Skipping", dest_file, ", it already exists")
        index += 1
        if limit_images_amount is not None and (index >= limit_images_amount):
            break
