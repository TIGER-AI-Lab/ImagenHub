from typing import List, Union, Tuple, Dict, Optional
import torch
from tqdm import tqdm
from PIL import Image

from imagen_hub.utils.null_inversion import NullInversion
from imagen_hub.pipelines.prompt2prompt.ptp_attn_control import make_controller
from imagen_hub.pipelines.prompt2prompt.ptp_utils import text2image

class Prompt2promptPipeline():
    """
    Prompt2promptPipeline
    """
    def __init__(self, sd_pipe, steps=50, guidance_scale=7.5, LOW_RESOURCE=False):
        self.LOW_RESOURCE = LOW_RESOURCE
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.pipe = sd_pipe
        self.device = sd_pipe.device

    def get_controller(self, prompts, source_subject_word=None, target_subject_word=None, is_replace_controller=True, cross_replace_steps = 0.3, self_replace_steps = 0.4, eq_params=None):
        """
        param:
            source_subject_word, target_subject_word : blend word for prompt2prompt. Eg. ["cat","tiger"]
            is_replace_controller : True for replacement (one word swap only), False for refinement (lengths of source and target prompts can be different). 
            eq_params : amplify attention to the word "____" by *k  e.g. eq_params={"words": ("____",), "values": (k,)}
        """
        blend_word = (((source_subject_word,), (target_subject_word,))) if target_subject_word!=None else None
        controller = make_controller(self.device, prompts=prompts, tokenizer=self.pipe.tokenizer,
                                     num_steps=self.steps, is_replace_controller=is_replace_controller,
                                     cross_replace_steps=cross_replace_steps,
                                     self_replace_steps=self_replace_steps,
                                     blend_words=blend_word, equilizer_params=eq_params)
        return controller
    
    def null_text_inverion(self, src_image, src_prompt):
        """
        invert real image into latent
        """
        null_inversion = NullInversion(self.pipe, ddim_steps=self.steps, guidance_scale=self.guidance_scale)
        x_t, uncond_embeddings = null_inversion.invert(src_image, src_prompt)
        return x_t, uncond_embeddings

    def generate_image(self, prompts, controller, generator, x_t, uncond_embeddings):
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