import os.path
import time
from typing import Dict, List
import numpy as np
import torch
import yaml
from PIL import Image
from omegaconf import OmegaConf

from imagen_hub.pipelines.freecontrol.utils import merge_sweep_config
from imagen_hub.pipelines.freecontrol import make_pipeline
from imagen_hub.pipelines.freecontrol.module.scheduler import CustomDDIMScheduler
from imagen_hub.pipelines.freecontrol.controlnet_processor import make_processor


def load_ckpt_pca_list(config_path='imagen_hub/pipelines/freecontrol/config/gradio_info.yaml'):
    """
    Load the checkpoint and pca basis list from the config file
    :param config_path:
    :return:
    models : Dict: The dictionary of the model checkpoints
    pca_basis_dict : List : The list of the pca basis

    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    # load from config
    with open(config_path, 'r') as f:
        gradio_config = yaml.safe_load(f)

    models: Dict = gradio_config['checkpoints']
    pca_basis_dict: Dict = dict()
    # remove non-exist model
    for model_version in list(models.keys()):
        for model_name in list(models[model_version].keys()):
            if "naive" not in model_name and not os.path.isfile(models[model_version][model_name]["path"]):
                models[model_version].pop(model_name)
            else:
                # Add the path of PCA basis to the pca_basis dict
                basis_dict = models[model_version][model_name]["pca_basis"]
                for key in list(basis_dict.keys()):
                    if not os.path.isfile(basis_dict[key]):
                        basis_dict.pop(key)
                if model_version not in pca_basis_dict.keys():
                    pca_basis_dict[model_version]: Dict = dict()
                if model_name not in pca_basis_dict[model_version].keys():
                    pca_basis_dict[model_version][model_name]: Dict = dict()
                pca_basis_dict[model_version][model_name].update(basis_dict)

    return models, pca_basis_dict


model_dict, pca_basis_dict = load_ckpt_pca_list()


def freecontrol_generate(condition_image, prompt, scale, ddim_steps, sd_version,
                         model_ckpt, pca_guidance_steps, pca_guidance_components,
                         pca_guidance_weight, pca_guidance_normalized,
                         pca_masked_tr, pca_guidance_penalty_factor, pca_warm_up_step, pca_texture_reg_tr,
                         pca_texture_reg_factor,
                         negative_prompt, seed, paired_objs,
                         pca_basis_dropdown, inversion_prompt, condition, img_size):
    control_type = condition

    if not control_type == "None":
        processor = make_processor(control_type.lower())
    else:
        processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x

    # get the config
    model_path = model_dict[sd_version][model_ckpt]['path']
    # define kwargs
    gradio_update_parameter = {
        # Stable Diffusion Generation Configuration ,
        'sd_config--guidance_scale': scale,
        'sd_config--steps': ddim_steps,
        'sd_config--seed': seed,
        'sd_config--dreambooth': False,
        'sd_config--prompt': prompt,
        'sd_config--negative_prompt': negative_prompt,
        'sd_config--obj_pairs': str(paired_objs),
        'sd_config--pca_paths': [pca_basis_dict[sd_version][model_ckpt][pca_basis_dropdown]],

        'data--inversion--prompt': inversion_prompt,
        'data--inversion--fixed_size': [img_size, img_size],

        # PCA Guidance Parameters
        'guidance--pca_guidance--end_step': int(pca_guidance_steps * ddim_steps),
        'guidance--pca_guidance--weight': pca_guidance_weight,
        'guidance--pca_guidance--structure_guidance--n_components': pca_guidance_components,
        'guidance--pca_guidance--structure_guidance--normalize': bool(pca_guidance_normalized),
        'guidance--pca_guidance--structure_guidance--mask_tr': pca_masked_tr,
        'guidance--pca_guidance--structure_guidance--penalty_factor': pca_guidance_penalty_factor,

        'guidance--pca_guidance--warm_up--apply': True if pca_warm_up_step > 0 else False,
        'guidance--pca_guidance--warm_up--end_step': int(pca_warm_up_step * ddim_steps),
        'guidance--pca_guidance--appearance_guidance--apply': True if pca_texture_reg_tr > 0 else False,
        'guidance--pca_guidance--appearance_guidance--tr': pca_texture_reg_tr,
        'guidance--pca_guidance--appearance_guidance--reg_factor': pca_texture_reg_factor,

        # Cross Attention Guidance Parameters
        'guidance--cross_attn--end_step': int(pca_guidance_steps * ddim_steps),
        'guidance--cross_attn--weight': 0,

    }

    input_config = gradio_update_parameter

    # Load base config
    base_config = yaml.load(open("imagen_hub/pipelines/freecontrol/config/base.yaml", "r"), Loader=yaml.FullLoader)
    # Update the Default config by gradio config
    config = merge_sweep_config(base_config=base_config, update=input_config)
    config = OmegaConf.create(config)

    # set the correct pipeline
    pipeline_name = "SDPipeline"

    pipeline = make_pipeline(pipeline_name,
                             model_path,
                             torch_dtype=torch.float16).to('cuda')
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

    # create a inversion config
    inversion_config = config.data.inversion

    # Processor the condition image
    img = processor(condition_image)
    # flip the color for the scribble and canny: black background to white background
    if control_type == "scribble" or control_type == "canny":
        img = Image.fromarray(255 - np.array(img))

    condition_image_latents = pipeline.invert(img=img, inversion_config=inversion_config)

    inverted_data = {"condition_input": [condition_image_latents], }

    g = torch.Generator()
    g.manual_seed(config.sd_config.seed)

    img_list = pipeline(prompt=config.sd_config.prompt,
                        negative_prompt=config.sd_config.negative_prompt,
                        num_inference_steps=config.sd_config.steps,
                        generator=g,
                        config=config,
                        inverted_data=inverted_data)[0]

    # Display the result：
    # if the control type is not none, then we display output_image_with_control
    # if the control type is none, then we display output_image
    return img_list[-1]


class freecontrol():
    def __init__(self, sd_version='1.5', model_ckpt='naive', pca_basis='cat_step_200_sample_20_id_0'):
        # Load checkpoint and pca basis list
        pca_basis_list = list(pca_basis_dict[sd_version][model_ckpt].keys()) if \
            pca_basis_dict[sd_version][model_ckpt].keys() is not None else []
        assert sd_version in list(model_dict.keys())  # it is an element in the list chosen by the user
        assert model_ckpt in list(model_dict[sd_version].keys())
        assert pca_basis in pca_basis_list

        self.sd_version = sd_version
        self.model_ckpt = model_ckpt
        self.pca_basis = pca_basis

    def infer_one_image(self, src_image: PIL.Image.Image = None,
                        prompt: str = None,
                        inversion_prompt: str = None,
                        paired_objects: str = None,
                        task: str = 'Canny',
                        seed: int = 42):
        """
        Infer one image using FreeControl.

        Args:
            src_image (PIL.Image.Image, optional): Source image in RGB format, pre-processed for guidance. Default is None.
            prompt (str, optional): prompt of the generating image. Default is None.
            inversion_prompt (str, optional): prompt of the provided image.
            paired_objects (str, optional): pair the object in inversion_prompt with the object in prompt that we want to correspond
            in form of "(obj from inversion prompt; obj from generation prompt)". Default is None.
            task (str, optional): Task type for FreeControl among
            ["None", "Scribble", "Depth", "Hed", "Seg", "Canny", "Normal", "Openpose"]. Defaults to "Canny".
            seed (int, optional): Seed for randomness. Default is 42.

        Returns:
            PIL.Image.Image: Processed image.
        """
        assert task in ["None", "Scribble", "Depth", "Hed", "Seg", "Canny", "Normal", "Openpose"]
        scale = 7.5
        ddim_steps = 200
        img_size = 512
        negative_prompt = None
        pca_guidance_steps = 0.6
        pca_guidance_components = 64
        pca_guidance_weight = 600
        pca_guidance_normalized = True
        pca_masked_tr = 0.3
        pca_guidance_penalty_factor = 10
        pca_warm_up_step = 0.05
        pca_texture_reg_tr = 0.5
        pca_texture_reg_factor = 0.1
        ips = [src_image, prompt, scale, ddim_steps, self.sd_version,
               self.model_ckpt, pca_guidance_steps, pca_guidance_components, pca_guidance_weight,
               pca_guidance_normalized,
               pca_masked_tr, pca_guidance_penalty_factor, pca_warm_up_step, pca_texture_reg_tr, pca_texture_reg_factor,
               negative_prompt, seed, paired_objects,
               self.pca_basis, inversion_prompt, task, img_size]
        img = freecontrol_generate(*ips)  # ips as arguments
        return img
