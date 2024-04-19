from typing import List, Union, Tuple, Dict, Optional
import torch
from tqdm import tqdm
from PIL import Image

from imagen_hub.pipelines.infedit.Unified_attention_ctrl import make_controller
from imagen_hub.pipelines.infedit.ptp_utils import register_attention_control

class InfEditPipeline:
    """
    A pipeline for transforming prompts and generating associated images.

    Attributes:
        LOW_RESOURCE (bool): Flag to indicate resource constraint.
        steps (int): Number of steps for inference.
        guidance_scale_s (float): Source scale for guidance during generation.
        guidance_scale_t (float): Target scale for guidance during generation.
        pipe (object): The underlying model pipeline.
        device (torch.device): Device on which the model operates.
        torch_dtype (torch.dtype): dtype of the pipe.
        encoder (object): Text encoder of the pipeline.
        tokenizer (object): Text tokenizer of the pipeline.
    """
    def __init__(self,sd_pipe, steps=12, guidance_scale_s=1, guidance_scale_t=2, LOW_RESOURCE=False,device = "cuda",torch_dtype =torch.float16  ):
        """
        Initializes the Prompt2promptPipeline.

        Args:
            sd_pipe (object): The underlying model pipeline.
            steps (int, optional): Number of steps for inference. Defaults to 50.
            guidance_scale_s (float): Source scale for guidance during generation. Defaults to 1.
            guidance_scale_t (float): Target scale for guidance during generation. Defaults to 2.
            LOW_RESOURCE (bool, optional): Flag to indicate resource constraint. Defaults to False.
            device (torch.device): Device on which the model operates. Defaults to cuda.
            torch_dtype (torch.dtype): dtype of the pipe. Defaults to torch.float16.
        """
        self.LOW_RESOURCE = LOW_RESOURCE
        self.steps = steps
        self.guidance_scale_s = guidance_scale_s
        self.guidance_scale_t = guidance_scale_t
        self.pipe = sd_pipe
        self.device = device
        self.torch_dtype = torch_dtype
        self.encoder = sd_pipe.text_encoder
        self.tokenizer = sd_pipe.tokenizer

    def get_controller(self, src_prompt, target_prompt, source_subject_word=None, target_subject_word=None,  denoise=False , strength=0.7,
               cross_replace_steps=0.7, self_replace_steps=0.7, thresh_e=0.3, thresh_m=0.3):
        """
        Retrieves the attention controller for the provided prompts.

        Args:
            prompts (List[str]): List of prompts to consider.
            source_subject_word (str, optional): Word in source prompt to be replaced.
            target_subject_word (str, optional): Word in target prompt replacing source_subject_word.
            denoise (bool, optional): Whether to use the denoise mode of the editing pipeline. Defaults to False
            strength (float, optional): Editin step portion in denoise mode if denoise is True. Defaults to 0.7.
            cross_replace_steps (float, optional): Steps for cross-replacement. Defaults to 0.7.
            self_replace_steps (float, optional): Steps for self-replacement. Defaults to 0.7.
            thresh_e (float, optional): threshhold for edited mask in local blend. Defaults to 0.3
            thresh_m (float, optional): threshhold for mutual mask in local blend. Defaults to 0.3

        Returns:
            object: Attention controller based on prompts.
        """
        self.strength=strength
        self.denoise=denoise
        blend_word = (((source_subject_word,), (target_subject_word,))) if target_subject_word!=None else None
        print(blend_word)
        controller = make_controller(self.device,self.torch_dtype, src_prompt, target_prompt, self.tokenizer,self.encoder, ""  , "",  num_inference_steps=self.steps, denoise=denoise , strength=strength,
               cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, thresh_e=thresh_e, thresh_m=thresh_m)
        return controller
    
    
    def generate_image(self,img, src_prompt, target_prompt, controller):
        """
        Generates an image based on the given prompts and controller.

        Args:
            prompts (List[str]): List of prompts for generation.
            img (PIL.Image): Source image.
            controller (object): Attention controller based on prompts.
            generator (object): Model for image generation.

        Returns:
            Image: Generated image.
        """
        register_attention_control(self.pipe, controller)
        if self.denoise is False:
            self.strength = 1
        print(src_prompt,target_prompt)
        results = self.pipe(prompt=target_prompt,
                    source_prompt=src_prompt,
                    positive_prompt="",
                    negative_prompt="",
                    image=img,
                    num_inference_steps=self.steps,
                    eta=1,
                    strength=self.strength,
                    guidance_scale=self.guidance_scale_t,
                    source_guidance_scale=self.guidance_scale_s,
                    denoise_model=self.denoise,
                    callback = controller.step_callback
                    )
        return results.images[0]