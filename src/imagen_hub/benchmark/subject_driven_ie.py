from imagen_hub.loader.benchmark_loader import dump_dataset_info, load_subject_driven_ie_dataset
from imagen_hub.utils.save_image_helper import save_pil_image, get_concat_pil_images
from imagen_hub import infermodels
from typing import Union, Optional, Callable
import os
from PIL import Image
from tqdm import tqdm

def infer_subject_driven_ie_bench(model,
                              result_folder: str = 'results',
                              experiment_name: str = "Exp_Subject-Driven_IE",
                              overwrite_model_outputs: bool = False,
                              overwrite_inputs: bool = False,
                              limit_images_amount: Optional[int] = None):
    """
    Run inference on the subject-driven image editing benchmark.

    This function processes a dataset for image editing tasks based on prompts and the specific
    subject-driven model provided. Depending on the model type, it uses pre-trained weights for
    the inference. The function supports resumption by checking and potentially overwriting existing
    model outputs and inputs.

    Args:
        model: The Subject-driven Image Editing model instance.
        result_folder (str, optional): Root directory where the results will be stored. Defaults to 'results'.
        experiment_name (str, optional): Name of the sub-directory inside 'result_folder' for the current experiment. Defaults to "Exp_Subject-Driven_IE".
        overwrite_model_outputs (bool, optional): Flag to indicate whether to overwrite existing model outputs. Useful for resuming runs. Defaults to False.
        overwrite_inputs (bool, optional): Flag to indicate whether to overwrite existing input samples. Usually set to False unless inputs need to be regenerated. Defaults to False.
        limit_images_amount (int, optional): Limit on the number of images to process. If None, all images in the dataset will be processed. Defaults to None.

    Returns:
        None. The edited images and other relevant files are saved to disk.

    Notes:
        The function processes each sample from the dataset, generates a unique ID for it, and saves
        the results in the respective directories. The results include the processed image and any
        associated metadata.
    """

    dataset, dataset_name = load_subject_driven_ie_dataset(with_name_att=True)
    dreambooth_data = dataset['train']
    eval_data = dataset['eval']

    def get_identifier_from_subject(subject: str):
        # TODO terrible way to get the identifier, refractor it later
        for data in dreambooth_data:
            if data['subject'] == subject:
                return data['identifier'], data['image']
        raise NotImplementedError("Special Token not found")

    def get_default_prompts(class_name: str, special_token: str):
        """
        class_name : str is the subject class name
        special_token : str
        """
        src_prompt = "photo of a " + class_name
        target_prompt = "photo of a " + special_token + " " + class_name
        return src_prompt, target_prompt

    def process_dataset_uid(sample):
        uid = sample['uid']
        return f"sample_{uid}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(eval_data,
                          required_attributes=['subject'],
                          uid_preprocess_fn=process_dataset_uid,
                          save_folder_path=os.path.join(
                              result_folder, experiment_name),
                          filename_wo_ext='dataset_lookup')

    print("========> Running Benchmark Dataset:", experiment_name, "| Model:", model.__class__.__name__)
    index = 0
    for sample in tqdm(eval_data):
        file_basename = process_dataset_uid(sample)
        dest_folder = os.path.join(result_folder, experiment_name, model.__class__.__name__)
        dest_file = os.path.join(dest_folder, file_basename)
        subject = sample['subject']
        subject_name = " ".join(subject.split('_'))
        subject_name = ''.join([i for i in subject_name if not i.isdigit()])
        special_token, subject_image = get_identifier_from_subject(subject)
        src_prompt, target_prompt = get_default_prompts(subject_name, special_token)

        if overwrite_model_outputs or not os.path.exists(dest_file):
            print("========> Inferencing", dest_file)

            sample_input = sample['image'].resize(
                (512, 512), Image.LANCZOS)

            if isinstance(model, infermodels.PhotoSwap):
                model.load_new_subject_weight(os.path.join("checkpoints", "DreamBooth-Models", subject),
                                              subject,
                                              special_token)
                output = model.infer_one_image(src_image=sample_input,
                                               src_prompt=src_prompt,
                                               target_prompt=target_prompt)
            if isinstance(model, infermodels.DreamEdit):
                weight_path = os.path.join("checkpoints", "last.ckpt")
                model.load_new_subject_weight(os.path.join("checkpoints", "DreamEdit-DreamBooth-Models", "dreamedit_official_ckpt", f"{subject}-{special_token}", weight_path),
                                              subject,
                                              special_token)
                output = model.infer_one_image(src_image=sample_input,
                                               src_prompt=src_prompt,
                                               target_prompt=target_prompt)
            if isinstance(model, infermodels.BLIPDiffusion_Edit):
                output = model.infer_one_image(src_image=sample_input,
                                cond_image=subject_image,
                                text_prompt="",
                                src_subject_name=subject_name,
                                cond_subject_name=subject_name,
                                target_subject_name=subject_name)

            output = output.resize((512, 512), Image.LANCZOS)
            save_pil_image(sample_input, os.path.join(
                result_folder, experiment_name, "input"), file_basename, overwrite=overwrite_inputs)
            save_pil_image(subject_image, os.path.join(
                result_folder, experiment_name, "token"), file_basename, overwrite=overwrite_inputs)
            save_pil_image(output, dest_folder, file_basename)
        else:
            print("========> Skipping", dest_file, ", it already exists")
        index += 1
        if limit_images_amount is not None and (index >= limit_images_amount):
            break
