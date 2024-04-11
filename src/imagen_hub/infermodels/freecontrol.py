import os.path
import time
from typing import Dict, List
import numpy as np
import torch
import yaml
import PIL
from PIL import Image
from omegaconf import OmegaConf


class FreeControl():
    """
    These codes and pipelines are adapted from https://github.com/genforce/freecontrol/tree/main
    """

    def __init__(self, device='cuda',
                 config_path='imagen_hub/pipelines/freecontrol/config/gradio_info.yaml',
                 sd_version='1.5',
                 model_ckpt='naive',
                 pca_basis='cat_step_200_sample_20_id_0',
                 scale=7.5,
                 ddim_steps=200,
                 img_size=512,
                 pca_guidance_steps=0.6,
                 pca_guidance_components=64,
                 pca_guidance_weight=600,
                 pca_guidance_normalized=True,
                 pca_masked_tr=0.3,
                 pca_guidance_penalty_factor=10,
                 pca_warm_up_step=0.05,
                 pca_texture_reg_tr=0.5,
                 pca_texture_reg_factor=0.1):
        from imagen_hub.pipelines.freecontrol.utils import merge_sweep_config
        from imagen_hub.pipelines.freecontrol.utils import load_ckpt_pca_list
        from imagen_hub.pipelines.freecontrol import make_pipeline
        from imagen_hub.pipelines.freecontrol.module.scheduler import CustomDDIMScheduler
        from imagen_hub.pipelines.freecontrol.controlnet_processor import make_processor
        self.merge_sweep_config = merge_sweep_config
        self.make_pipeline = make_pipeline
        self.CustomDDIMScheduler = CustomDDIMScheduler
        self.make_processor = make_processor
        
        self.device = device

        # Load checkpoint and pca basis list
        self.config_path = config_path
        model_dict, pca_basis_dict = load_ckpt_pca_list(config_path)
        pca_basis_list = list(pca_basis_dict[sd_version][model_ckpt].keys()) if \
            pca_basis_dict[sd_version][model_ckpt].keys() is not None else []

        assert sd_version in list(model_dict.keys())  # it is an element in the list chosen by the user
        assert model_ckpt in list(model_dict[sd_version].keys())
        assert pca_basis in pca_basis_list

        self.sd_version = sd_version
        self.model_ckpt = model_ckpt
        self.pca_basis = pca_basis

        # get the config
        self.model_path = model_dict[sd_version][model_ckpt]['path']

        self.tasks = {"control_none": "none",
                      "control_hed": "hed",
                      "control_canny": "canny",
                      "control_seg": "seg",
                      "control_depth": "depth",
                      "control_normal": "normal",
                      "control_mlsd": "mlsd",
                      "control_openpose": "openpose",
                      "control_scribble": "scribble"}

        # Load base config
        self.base_config = yaml.load(open("imagen_hub/pipelines/freecontrol/config/base.yaml", "r"),
                                     Loader=yaml.FullLoader)
        # define kwargs
        self.input_config = {
            # Stable Diffusion Generation Configuration ,
            'sd_config--guidance_scale': scale,
            'sd_config--steps': ddim_steps,
            'sd_config--dreambooth': False,
            'sd_config--pca_paths': [pca_basis_dict[sd_version][model_ckpt][pca_basis]],

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

    def infer_one_image(self, src_image: PIL.Image.Image = None, prompt: str = None, task: str = "control_canny",
                        seed: int = 42):
        """
        Infer one image using FreeControl.

        Args:
            src_image (PIL.Image.Image, optional): Source image in RGB format, pre-processed for guidance. Default is None.
            prompt (str, optional): prompt of the generating image. Default is None.
            task (str, optional): Task type for FreeControl among
            ["None", "Scribble", "Depth", "Hed", "Seg", "Canny", "Normal", "Openpose"]. Defaults to "Canny".
            seed (int, optional): Seed for randomness. Default is 42.

        Returns:
            PIL.Image.Image: Processed image.
        """

        assert task in self.tasks
        control_type = task

        if not control_type == "control_none":
            processor = self.make_processor(self.tasks[control_type])
        else:
            processor = lambda x: Image.open(x).convert("RGB") if type(x) == str else x

        self.input_config['sd_config--prompt'] = prompt
        self.input_config['sd_config--seed'] = seed

        # Update the Default config by gradio config
        config = self.merge_sweep_config(base_config=self.base_config, update=self.input_config)
        config = OmegaConf.create(config)

        # set the correct pipeline
        pipeline_name = "SDPipeline"

        pipeline = self.make_pipeline(pipeline_name,
                                 self.model_path,
                                 torch_dtype=torch.float16).to(self.device)
        pipeline.scheduler = self.CustomDDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")

        # create a inversion config
        inversion_config = config.data.inversion

        # Processor the condition image
        img = processor(src_image)
        # flip the color for the scribble and canny: black background to white background
        if control_type == "control_scribble" or control_type == "control_canny":
            img = Image.fromarray(255 - np.array(img))

        condition_image_latents = pipeline.invert(img=img, inversion_config=inversion_config)

        inverted_data = {"condition_input": [condition_image_latents], }

        g = torch.Generator()
        g.manual_seed(config.sd_config.seed)

        img_list = pipeline(use_pca=False,
                            prompt=config.sd_config.prompt,
                            negative_prompt=config.sd_config.negative_prompt,
                            num_inference_steps=config.sd_config.steps,
                            generator=g,
                            config=config,
                            inverted_data=inverted_data)[0]

        # Display the result：
        # if the control type is not none, then we display output_image_with_control
        # if the control type is none, then we display output_image
        return img_list[0]
