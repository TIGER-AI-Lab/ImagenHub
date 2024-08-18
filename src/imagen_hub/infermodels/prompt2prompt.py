import torch
import PIL

from diffusers import StableDiffusionPipeline, DDIMScheduler

class Prompt2prompt():
    """
    A class for Prompt-to-Prompt.
    
    References: https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb
    """

    def __init__(self, device="cuda", weight="CompVis/stable-diffusion-v1-4", src_subject_word=None, target_subject_word=None):
        """
        Initialize the Prompt2prompt class.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            weight (str, optional): Pretrained model weight. Defaults to "CompVis/stable-diffusion-v1-4".
            src_subject_word (str, optional): Source subject word. Defaults to None.
            target_subject_word (str, optional): Target subject word. Defaults to None.
        """
        from imagen_hub.pipelines.prompt2prompt.pipeline_ptp import Prompt2promptPipeline

        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(weight,
                                                            safety_checker=None)
        self.pipe.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        try:
            self.pipe.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        self.ptp_pipe = Prompt2promptPipeline(
            self.pipe, steps=50, guidance_scale=7.5, LOW_RESOURCE=False)

        self.src_subject_word = src_subject_word
        self.target_subject_word = target_subject_word

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, is_replace_controller: bool = False, replace_steps=[0.3, 0.3], eq_params=None, seed: int = 42, num_inner_steps=10):
        """
        Perform inference on a source image based on provided prompts.

        Args:
            src_image (PIL.Image): Source image.
            src_prompt (str, optional): Description or caption of the source image.
            target_prompt (str, optional): Desired description or caption for the output image.
            instruct_prompt (str, optional): Instruction prompt. Not utilized in this implementation.
            is_replace_controller (bool, optional): Indicator whether it's a replacement. True for replacement (one word swap only), False for refinement (lengths of source and target prompts can be different). Defaults to False.
            replace_steps (list of float, optional): A list of [cross_replace_steps, self_replace_steps]. Defaults to [0.3, 0.3].
            eq_params (dict, optional): Parameters to amplify attention to specific words. For example, eq_params={"words": ("XXXXX",), "values": (k,)} amplifies attention to the word "XXXXX" by times k. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            PIL.Image: Transformed image based on the provided prompts.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.Generator(self.device).manual_seed(seed)
        x_t, uncond_embeddings = self.ptp_pipe.null_text_inverion(
            src_image, src_prompt, num_inner_steps=num_inner_steps)
        prompts = [src_prompt, target_prompt]
        controller = self.ptp_pipe.get_controller(prompts,
                                                  self.src_subject_word,
                                                  self.target_subject_word,
                                                  is_replace_controller=is_replace_controller,
                                                  cross_replace_steps=replace_steps[0],
                                                  self_replace_steps=replace_steps[1],
                                                  eq_params=eq_params)
        image = self.ptp_pipe.generate_image(prompts,
                                             controller,
                                             generator,
                                             x_t,
                                             uncond_embeddings)
        return image
