from typing import List, Union, Tuple, Dict, Optional
import torch
from tqdm import tqdm
from PIL import Image

from imagen_hub.utils.null_inversion import NullInversion
from imagen_hub.pipelines.photoswap.swapping_class import AttentionSwap
from imagen_hub.pipelines.prompt2prompt.ptp_utils import text2image


class PhotoswapPipeline():
    """
    Provides a pipeline to handle photo swapping using AttentionSwap.

    Attributes:
        LOW_RESOURCE (bool): Flag indicating resource mode.
        steps (int): Number of steps for inversion and image generation.
        guidance_scale (float): Scale factor for guiding the process.
        pipe (object): Predefined pipeline for processing.
        device (str): Device the pipeline is running on (e.g., 'cuda' or 'cpu').

    Args:
        sd_pipe (object): The pipeline for processing.
        steps (int, optional): Number of steps. Defaults to 50.
        guidance_scale (float, optional): Guidance scale factor. Defaults to 7.5.
        LOW_RESOURCE (bool, optional): Flag to indicate resource mode. Defaults to False.
    """

    def __init__(self, sd_pipe, steps=50, guidance_scale=7.5, LOW_RESOURCE=False):
        self.LOW_RESOURCE = LOW_RESOURCE
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.pipe = sd_pipe
        self.device = sd_pipe.device

    def get_controller(self, prompts, source_subject_word=None, target_subject_word=None, cross_map_replace_steps=0.3, self_output_replace_steps=0.4, self_map_replace_steps=0.0):
        """
        Returns a controller based on the provided parameters.

        Args:
            prompts (List[str]): List of prompts.
            source_subject_word (str, optional): Source subject word for swapping.
            target_subject_word (str, optional): Target subject word for swapping.
            cross_map_replace_steps (float, optional): Cross map replacement steps. Defaults to 0.3.
            self_output_replace_steps (float, optional): Self output replacement steps. Defaults to 0.4.
            self_map_replace_steps (float, optional): Self map replacement steps. Defaults to 0.0.

        Returns:
            AttentionSwap: Instance of the AttentionSwap class.
        
        Raises:
            AssertionError: If the sum of self_output_replace_steps and self_map_replace_steps is greater than 1.0.

        Note:
            Add another word 'woman' so the source and target prompt have the same token length. This is not a necessary step.
            source_subject_word = 'woman'
            source_prompt = "A photo of a woman woman, smiling"
            Changing the subject into the target. 'sks woman' is the name used when training this example concept learning model.
            target_subject_word = 'sks'
            target_prompt = "A photo of a sks woman, smiling"
            The three parameters cross_map_replace_steps, self_output_replace_steps, self_map_replace_steps
            are used to change the tuning. Different concept learning model may have different parameter range.
        """
        assert self_output_replace_steps + self_map_replace_steps <= 1.0
        controller = AttentionSwap(prompts, self.steps, cross_map_replace_steps=cross_map_replace_steps, self_map_replace_steps=self_map_replace_steps,
                                   self_output_replace_steps=self_output_replace_steps, source_subject_word=source_subject_word, target_subject_word=target_subject_word,
                                   tokenizer=self.pipe.tokenizer, device=self.device, LOW_RESOURCE=self.LOW_RESOURCE)
        return controller

    def null_text_inverion(self, src_image, src_prompt):
        """
        Inverts a real image into latent representation.

        Args:
            src_image (Image): The source image to be inverted.
            src_prompt (str): The associated prompt for the image.

        Returns:
            Tuple: Tuple containing the latent representation and unconditional embeddings.
        """
        null_inversion = NullInversion(
            self.pipe, ddim_steps=self.steps, guidance_scale=self.guidance_scale)
        x_t, uncond_embeddings = null_inversion.invert(src_image, src_prompt)
        return x_t, uncond_embeddings

    def generate_image(self, prompts, controller, generator, x_t, uncond_embeddings):
        """
        Generates an image based on the provided parameters.

        Args:
            prompts (List[str]): List of prompts for image generation.
            controller (AttentionSwap): Instance of the AttentionSwap class.
            generator (object): An image generator.
            x_t (torch.Tensor): Latent representation tensor.
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
