from typing import List, Union, Tuple, Dict, Optional
import torch
from tqdm import tqdm
from PIL import Image

from imagen_hub.utils.null_inversion import NullInversion
from imagen_hub.pipelines.prompt2prompt.ptp_attn_control import make_controller
from imagen_hub.pipelines.prompt2prompt.ptp_utils import text2image

class Prompt2promptPipeline():
    """
    A pipeline for transforming prompts and generating associated images.

    Attributes:
        LOW_RESOURCE (bool): Flag to indicate resource constraint.
        steps (int): Number of steps for inference.
        guidance_scale (float): Scale for guidance during generation.
        pipe (object): The underlying model pipeline.
        device (torch.device): Device on which the model operates.
    """
    def __init__(self, sd_pipe, steps=50, guidance_scale=7.5, LOW_RESOURCE=False):
        """
        Initializes the Prompt2promptPipeline.

        Args:
            sd_pipe (object): The underlying model pipeline.
            steps (int, optional): Number of steps for inference. Defaults to 50.
            guidance_scale (float, optional): Scale for guidance during generation. Defaults to 7.5.
            LOW_RESOURCE (bool, optional): Flag to indicate resource constraint. Defaults to False.
        """
        self.LOW_RESOURCE = LOW_RESOURCE
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.pipe = sd_pipe
        self.device = sd_pipe.device

    def get_controller(self, prompts, source_subject_word=None, target_subject_word=None, is_replace_controller=True, cross_replace_steps = 0.3, self_replace_steps = 0.4, eq_params=None):
        """
        Retrieves the attention controller for the provided prompts.

        Args:
            prompts (List[str]): List of prompts to consider.
            source_subject_word (str, optional): Word in source prompt to be replaced.
            target_subject_word (str, optional): Word in target prompt replacing source_subject_word.
            is_replace_controller (bool, optional): True for replacement, False for refinement. Defaults to True.
            cross_replace_steps (float, optional): Steps for cross-replacement. Defaults to 0.3.
            self_replace_steps (float, optional): Steps for self-replacement. Defaults to 0.4.
            eq_params (Dict, optional): Parameters to amplify attention to specific words.

        Returns:
            object: Attention controller based on prompts.
        """
        blend_word = (((source_subject_word,), (target_subject_word,))) if target_subject_word!=None else None
        print(blend_word)
        controller = make_controller(self.device, prompts=prompts, tokenizer=self.pipe.tokenizer,
                                     num_steps=self.steps, is_replace_controller=is_replace_controller,
                                     cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps,
                                     blend_words=blend_word, equilizer_params=eq_params)
        return controller
    
    def null_text_inverion(self, src_image, src_prompt, num_inner_steps=10):
        """
        Inverts a real image to its latent representation using the provided prompt.

        Args:
            src_image (Image): Source image to invert.
            src_prompt (str): Text prompt corresponding to the source image.

        Returns:
            Tuple: Tuple containing latent representation and embeddings.
        """
        null_inversion = NullInversion(self.pipe, ddim_steps=self.steps, guidance_scale=self.guidance_scale)
        x_t, uncond_embeddings = null_inversion.invert(src_image, src_prompt, num_inner_steps=num_inner_steps)
        return x_t, uncond_embeddings

    def generate_image(self, prompts, controller, generator, x_t, uncond_embeddings):
        """
        Generates an image based on the given prompts and controller.

        Args:
            prompts (List[str]): List of prompts for generation.
            controller (object): Attention controller based on prompts.
            generator (object): Model for image generation.
            x_t (torch.Tensor): Latent tensor.
            uncond_embeddings (torch.Tensor): Unconditional embeddings tensor.

        Returns:
            Image: Generated image.
        """
        images, x_t = text2image(
                self.pipe,
                prompts,
                controller,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                latent=x_t,
                uncond_embeddings=uncond_embeddings,
                start_time=50,
                return_type='image'
        )
        return Image.fromarray(images[1])
