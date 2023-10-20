from imagen_hub.loader.benchmark_loader import dump_dataset_info, load_multi_concept_ic_dataset
from imagen_hub.utils.save_image_helper import save_pil_image, get_concat_pil_images
from imagen_hub import infermodels
from typing import Union, Optional, Callable
import os
from PIL import Image
from tqdm import tqdm
import torch

def infer_multi_concept_ic_bench(model,
                        result_folder: str = 'results',
                        experiment_name: str = "Exp_Multi-Concept_IC",
                        overwrite_model_outputs: bool = False,
                        overwrite_inputs: bool = False,
                        limit_images_amount: Optional[int] = None):
    """
        Args:
            model: The Multiple Subject Image Generation model.
            result_folder: result root folder
            experiment_name: folder name in the result root folder for storing experiment
            task_id: task type
            overwrite_model_outputs: bool to indicate overwriting existing files or not. Designed for resuming the run
            overwrite_inputs: bool to indicate overwriting input sample files or not. Usually set to False
    """

    # test only code for now, need to move train here
    dataset, dataset_name = load_multi_concept_ic_dataset(with_name_att=True)
    data = dataset['test']

    def process_dataset_uid(sample):
        uid = sample['uid']
        return f"sample_{uid}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(data,
                          required_attributes=['prompt', 'concept1', 'concept2'],
                          uid_preprocess_fn=process_dataset_uid,
                          save_folder_path=os.path.join(
                              result_folder, experiment_name),
                          filename_wo_ext='dataset_lookup')

    input_folder = os.path.join(result_folder, 'input')
    if overwrite_inputs or not os.path.exists(input_folder):
        trains = {}
        for sample in tqdm(dataset['train']):
            image = sample['image']
            c = sample['concept']
            if c not in trains:
                trains[c] = []
            trains[c].append(image)
        print(f"=======> Dumping dataset input")
        for sample in tqdm(dataset['test']):
            file_basename = process_dataset_uid(sample)
            prompt = sample['prompt']
            c1 = sample['concept1']
            c2 = sample['concept2']
            #uid = sample['uid']
            newsize = (512, 512)
            concept_images = get_concat_pil_images([trains[c1][0].resize(newsize), trains[c2][0].resize(newsize)])
            save_pil_image(concept_images, input_folder, file_basename)

    print("========> Running Benchmark Dataset:", experiment_name, "| Model:", model.__class__.__name__)


    tests = {}
    for sample in tqdm(dataset['test']):
        prompt = sample['prompt']
        c1 = sample['concept1']
        c2 = sample['concept2']
        #uid = sample['uid']
        name = f'{c1}+{c2}'
        file_basename = process_dataset_uid(sample)
        if not name in tests:
            tests[name] = []
        tests[name].append((prompt, file_basename))
    for name, test in tqdm(tests.items()):
        c1, c2 = name.split('+')
        c1_clean = c1.replace(' ', '_')
        c2_clean = c2.replace(' ', '_')

        if isinstance(model, infermodels.DreamBoothMulti):
            # Setup Model
            data_dir = 'temp_data' # not relevent atm
            trained_model_dir = os.path.join("checkpoints", "ImagenHub_Multi-Concept_IC", "dreambooth", f"{c1_clean}+{c2_clean}")
            model.set_pipe([c1, c2],
                            ["<new1>", "<new2>"],
                            data_dir,
                            output_dir=trained_model_dir)
        if isinstance(model, infermodels.CustomDiffusion):
            # Setup Model
            data_dir = os.path.join('temp_data', 'instance', 'data') # not relevent atm
            model_folder = os.path.join("checkpoints", "ImagenHub_Multi-Concept_IC", "custom-diffusion")
            model.set_pipe(c1,
                            c2,
                            os.path.join(data_dir, c1),
                            os.path.join(data_dir, c2),
                            model_folder=model_folder)
            ckpt_name = c1_clean + '+' + c2_clean + '-sdv4'
            all_ckpt_names = os.listdir(model_folder)
            ckpt_name = [k for k in all_ckpt_names if ckpt_name in k][0]
        if isinstance(model, infermodels.TextualInversionMulti):
            # Setup Model
            model_folder = os.path.join("checkpoints", "ImagenHub_Multi-Concept_IC", "textual_inversion")
            paths = os.listdir(model_folder)
            c1_holder = [p for p in paths if c1_clean in p][0].split('+')[1]
            c2_holder = [p for p in paths if c2_clean in p][0].split('+')[1]
            paths = [os.path.join(model_folder, p) for p in [f'{c1_clean}+{c1_holder}', f'{c2_clean}+{c2_holder}']]
            model.set_pipe(paths)

        index = 0
        for prompt, file_basename in test:
            output_dir = os.path.join('results', model.__class__.__name__)

            if overwrite_model_outputs or not os.path.exists(os.path.join(output_dir, file_basename)):
                if isinstance(model, infermodels.DreamBoothMulti):
                    # Infer
                    prompt = prompt.replace(c1, f'<new1> {c1}').replace(c2, f'<new2> {c2}')
                    print(prompt, c1, c2, trained_model_dir)
                    image = model.infer_one_image(trained_model_dir, prompt)

                if isinstance(model, infermodels.CustomDiffusion):
                    # Infer
                    prompt = prompt.replace(c1, f'<new1> {c1}').replace(c2, f'<new2> {c2}')
                    delta_ckpt = os.path.join(model_folder, ckpt_name, 'checkpoints', 'delta_epoch=000004.ckpt')
                    sd_ckpt = os.path.join(model_folder, 'sd-v1-4.ckpt')
                    image = model.infer_one_image(prompt,
                                                  delta_ckpt=delta_ckpt,
                                                  pretrained_ckpt=sd_ckpt)

                if isinstance(model, infermodels.TextualInversionMulti):
                    # Infer
                    prompt = prompt.replace(c1, f'{c1_holder} {c1}').replace(c2, f'{c2_holder} {c2}')
                    image = model.infer_one_image(prompt)

                save_pil_image(image, output_dir, file_basename)
                index += 1
                if limit_images_amount is not None and (index >= limit_images_amount):
                    break
