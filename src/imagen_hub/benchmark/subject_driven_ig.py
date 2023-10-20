from imagen_hub.loader.benchmark_loader import dump_dataset_info, load_subject_driven_ig_dataset
from imagen_hub.utils.save_image_helper import save_pil_image, get_concat_pil_images
from imagen_hub import infermodels
from typing import Union, Optional, Callable
import os
from PIL import Image
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def infer_subject_driven_ig_bench(model,
                                  result_folder: str = 'results',
                                  experiment_name: str = "Exp_Subject-Driven_IG",
                                  overwrite_model_outputs: bool = False,
                                  overwrite_inputs: bool = False,
                                  limit_images_amount: Optional[int] = None):
    """
    Run inference on the subject-driven image generation benchmark.

    This function processes a dataset for image generation based on prompts.
    Depending on the model type, it either trains the model on-the-fly or uses
    pre-trained weights. The function supports resumption by checking and
    potentially overwriting existing model outputs and inputs.

    Args:
        model: The Subject-driven Image Generation model instance.
        result_folder (str, optional): Root directory where the results will be stored. Defaults to 'results'.
        experiment_name (str, optional): Name of the sub-directory inside 'result_folder' for the current experiment. Defaults to "Exp_Subject-Driven_IG".
        overwrite_model_outputs (bool, optional): Flag to indicate whether to overwrite existing model outputs. Useful for resuming runs. Defaults to False.
        overwrite_inputs (bool, optional): Flag to indicate whether to overwrite existing input samples. Usually set to False unless inputs need to be regenerated. Defaults to False.
        limit_images_amount (int, optional): Limit on the number of images to process. If None, all images in the dataset will be processed. Defaults to None.

    Returns:
        None. The generated images and other relevant files are saved to disk.

    Notes:
        The function processes each sample from the dataset, generates a unique ID for it, and saves
        the results in the respective directories. The results include the processed image and any
        associated metadata.
    """
    dataset, dataset_name = load_subject_driven_ig_dataset(with_name_att=True)
    dreambooth_data = dataset['train']
    eval_data = dataset['eval']

    def get_identifier_from_subject(subject: str):
        # TODO terrible way to get the identifier, refractor it later
        for data in dreambooth_data:
            if data['subject'] == subject:
                return data['identifier'], data['image']
        raise NotImplementedError("Special Token not found")

    def process_dataset_uid(sample):
        uid = sample['uid']
        #subject = sample['subject']
        return f"sample_{uid}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(eval_data,
                          required_attributes=['subject_id', 'subject', 'prompt'],
                          uid_preprocess_fn=process_dataset_uid,
                          save_folder_path=os.path.join(result_folder, experiment_name),
                          filename_wo_ext='dataset_lookup')

    input_folder = os.path.join(result_folder, experiment_name, 'input')
    if overwrite_inputs or not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)

    print("========> Running Benchmark Dataset:", experiment_name, "| Model:", model.__class__.__name__)

    index = 0
    for sample in tqdm(eval_data):
        file_basename = process_dataset_uid(sample)
        dest_folder = os.path.join(result_folder, experiment_name, model.__class__.__name__)
        dest_file = os.path.join(dest_folder, file_basename)
        special_token = 'sks'
        subject = sample['subject']
        subject_name = " ".join(subject.split('_'))
        subject_name = ''.join([i for i in subject_name if not i.isdigit()])

        temp_dreambooth_data_path = os.path.join('data', 'DreamBooth', subject)
        class_dir = os.path.join(temp_dreambooth_data_path, "class")
        instance_dir = os.path.join(temp_dreambooth_data_path, "instance")
        if not os.path.exists(temp_dreambooth_data_path) or not os.path.exists(class_dir):
            os.makedirs(temp_dreambooth_data_path, exist_ok=True)
            os.makedirs(class_dir, exist_ok=True)
            os.makedirs(instance_dir, exist_ok=True)

        # If no data is in the instance directory, populate it from the dreambooth dataset.
        if len(os.listdir(instance_dir)) == 0:
            count = 0
            for subject_sample in dreambooth_data:
                if subject_sample['subject'] == subject:
                    special_token = subject_sample['identifier']
                    save_pil_image(subject_sample['image'], instance_dir, "0"+ str(count), filetype="jpg")
                    count += 1
        else:
            # Get the special token for the subject.
            for subject_sample in dreambooth_data:
                if subject_sample['subject'] == subject:
                    special_token = subject_sample['identifier']
                    break

        input_file = os.path.join(input_folder, file_basename)
        if overwrite_inputs or not os.path.exists(input_file):
            input_image_path = os.path.join(instance_dir, os.listdir(instance_dir)[0])
            input_image = Image.open(input_image_path)
            save_pil_image(input_image, input_folder, file_basename)

        if overwrite_model_outputs or not os.path.exists(dest_file):
            print("========> Inferencing", dest_file)

            #Weights needed for some models.
            weight_path = os.path.join("checkpoints", experiment_name, model.__name__, subject)

            if isinstance(model, infermodels.BLIPDiffusion_Gen):
                _, cond_image = get_identifier_from_subject(subject)
                prompt = sample['prompt'].replace('<token>', '')  # replace with nothing
                output = model.infer_one_image(cond_image=cond_image,
                                                text_prompt=prompt,
                                                cond_subject_name=subject_name,
                                                target_subject_name=subject_name)
                save_pil_image(output, dest_folder=dest_folder, filename=file_basename)

            if isinstance(model, infermodels.DreamBooth) or isinstance(model, infermodels.DreamBoothLora):

                # Train
                if not os.path.exists(weight_path) or len(os.listdir(weight_path)) == 0:
                    print(f'not found checkpoint {weight_path}. Perform training now:')
                    os.makedirs(weight_path, exist_ok=True)
                    model.set_pipe(subject_name=subject_name,
                                    identifier=special_token,
                                    data_path=temp_dreambooth_data_path,
                                    output_dir=weight_path)
                    model.train()

                # Infer
                model.set_pipe(subject_name=subject_name,
                                identifier=special_token,
                                data_path=temp_dreambooth_data_path,
                                output_dir=weight_path)
                prompt = sample['prompt'].replace('<token>', special_token)
                print("prompt: ", prompt)
                image = model.infer_one_image(model_path=weight_path, instruct_prompt=prompt)

            if isinstance(model, infermodels.TextualInversion):

                # Train
                if not os.path.exists(weight_path) or len(os.listdir(weight_path)) == 0:
                    print(f'not found checkpoint {weight_path}. Perform training now:')
                    os.makedirs(weight_path, exist_ok=True)
                    init_token = word_tokenize(subject_name)
                    model.set_pipe(what_to_teach='object',
                            placeholder_token=special_token,
                            initializer_token=init_token[0],
                            output_dir=weight_path)

                    model.train(instance_dir)

                # Infer
                model.set_pipe(what_to_teach='object',
                        placeholder_token=special_token,
                        initializer_token=subject_name.split()[-1],
                        output_dir=weight_path)
                prompt = sample['prompt'].replace('<token> ' + subject_name, special_token)
                print("prompt: ", prompt)
                image = model.infer_one_image(output_dir=weight_path, prompt=prompt)

            save_pil_image(image, dest_folder, file_basename)

        index += 1
        if limit_images_amount is not None and (index >= limit_images_amount):
            break
