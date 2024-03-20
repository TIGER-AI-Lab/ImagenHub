import torch
import pytorch_lightning as pl
import PIL
import numpy as np
import einops
from pytorch_lightning import seed_everything
from numpy import asarray
import cv2
import os

from imagen_hub.pipelines.unicontrol import cldm_v15_unicontrol_yaml
from imagen_hub.pipelines.unicontrol.utils import check_safety

class UniControl():
    """
    UniControl pipeline for controlling the image generation process with a specific task.
    """
    def __init__(self,
                 device="cuda",
                 weight=None,
                 config=cldm_v15_unicontrol_yaml):
        """
        Initializes the UniControl instance.

        Args:
            device (str, optional): Device to use for inference ("cuda" or "cpu"). Defaults to "cuda".
            weight (str, optional): Path to model weights. If None, will download from the default URL.
            config (dict): Configuration for creating the model.
        """

        from imagen_hub.pipelines.unicontrol.cldm.model import create_model, load_state_dict
        from imagen_hub.pipelines.unicontrol.cldm.ddim_unicontrol_hacked import DDIMSampler
        from imagen_hub.utils.file_helper import get_file_path, download_weights_to_directory

        # Configs
        if weight is None:
            weight = download_weights_to_directory(url="https://storage.googleapis.com/sfr-unicontrol-data-research/unicontrol.ckpt",
                                                   save_dir=os.path.join("checkpoints", "ImagenHub_Control-Guided_IG", "UniControl"),
                                                   filename="unicontrol.ckpt")

        self.checkpoint_path = weight
        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        self.model = create_model(config).cpu()
        self.model.load_state_dict(load_state_dict(self.checkpoint_path, location='cpu'), strict=False) #, strict=False
        self.num_samples = 1
        self.guess_mode = False
        self.ddim_steps = 50
        self.scale = 9.0
        self.ddim_sampler = DDIMSampler(self.model)
        self.task_to_instruction = {"control_hed": "hed edge to image", "control_canny": "canny edge to image", "control_seg": "segmentation map to image", "control_depth": "depth map to image", "control_normal": "normal surface map to image", "control_img": "image editing", "control_openpose": "human pose skeleton to image", "control_hedsketch": "sketch to image", "control_bbox": "bounding box to image", "control_outpainting": "image outpainting", "control_grayscale": "gray image to color image", "control_blur": "deblur image to clean image", "control_inpainting": "image inpainting"}
        self.model.to(device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, prompt: str = None, task: str = "control_canny", seed: int = 42):
        """
        Generates an image using the UniControl model based on a given source image, prompt, and task.

        Args:
            src_image (PIL.Image, optional): Pre-processed source image for guidance. Defaults to None.
            prompt (str, optional): Textual prompt for additional guidance. Defaults to None.
            task (str, optional): The specific task for controlling the image generation process. Defaults to "control_canny".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            PIL.Image: The generated image.
        """
        seed_everything(seed)
        # configs from https://github.com/salesforce/UniControl

        hint = src_image
        hint = cv2.cvtColor(np.array(hint), cv2.COLOR_RGB2BGR)
        hint = hint.astype(np.float32) / 255.0
        hint = torch.from_numpy(hint)
        control = hint.cuda()
        H, W, C = control.shape

        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        task_dic = {}
        task_dic['name'] = task
        task_instruction = self.task_to_instruction[task]
        task_dic['feature'] = self.model.get_learned_conditioning(task_instruction)[:,:1,:]

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt] * self.num_samples)], "task": task_dic}
        un_cond = {"c_concat": [torch.zeros_like(control)] if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([""] * self.num_samples)]}
        shape = (4, H // 8, W // 8)

        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.num_samples,
                                                     shape, cond, verbose=False, eta=0,
                                                     unconditional_guidance_scale=self.scale,
                                                     unconditional_conditioning=un_cond)
        x_samples = self.model.decode_first_stage(samples)

        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

        x_checked_image, has_nsfw_concept = check_safety(x_samples)
        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
        for x_sample in x_checked_image_torch:
            x_sample = 255. * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = PIL.Image.fromarray(x_sample.astype(np.uint8))
        return img
