import torch
import PIL

from diffusers import StableDiffusionPipeline, DDIMScheduler


class PhotoSwap():
    """
    Wrapper for the PhotoSwap image transformation model.

    Reference: https://github.com/eric-ai-lab/photoswap/blob/main/real-image-swap.ipynb
    """
    def __init__(self, device="cuda", weight="CompVis/stable-diffusion-v1-4", src_subject_word=None, target_subject_word=None):
        """
        Initialize the PhotoSwap class.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            weight (str, optional): Model weights to load. Defaults to "CompVis/stable-diffusion-v1-4".
            src_subject_word (str, optional): Source subject word. Defaults to None.
            target_subject_word (str, optional): Target subject word (the special token e.g. sks). Defaults to None.
        """
        from imagen_hub.pipelines.photoswap.pipeline_photoswap import PhotoswapPipeline

        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(weight,
                                                            safety_checker=None).to(self.device)
        self.pipe.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        try:
            self.pipe.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")

        self.src_subject_word = src_subject_word
        self.target_subject_word = target_subject_word
        self.photoswap_pipe = PhotoswapPipeline(
            self.pipe, steps=50, guidance_scale=7.5, LOW_RESOURCE=False)

    def infer_one_image(self,
                        src_image: PIL.Image.Image,
                        src_prompt: str,
                        target_prompt: str,
                        replace_steps: list = [0.3, 0.3, 0],
                        seed: int = 42):
        """
        Modify the source image based on provided prompts.

        Args:
            src_image (Image): Source image.
            src_prompt (str): Source image caption.
            target_prompt (str): Target image caption.
            replace_steps (list, optional): List of replace steps. Defaults to [0.3, 0.3, 0].
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            Image: Modified image.
        """
        src_image = src_image.convert('RGB')  # force it to RGB format
        generator = torch.Generator(self.device).manual_seed(seed)
        x_t, uncond_embeddings = self.photoswap_pipe.null_text_inverion(
            src_image, src_prompt)
        prompts = [src_prompt, target_prompt]
        controller = self.photoswap_pipe.get_controller(prompts,
                                                        self.src_subject_word,
                                                        self.target_subject_word,
                                                        cross_map_replace_steps=replace_steps[0],
                                                        self_output_replace_steps=replace_steps[1],
                                                        self_map_replace_steps=replace_steps[2])
        image = self.photoswap_pipe.generate_image(prompts,
                                                   controller,
                                                   generator,
                                                   x_t,
                                                   uncond_embeddings)
        return image

    def load_new_subject_weight(self, weight, src_subject_word, target_subject_word):
        """
        Load new diffuser weights and set new subjects.

        Args:
            weight (str): Model weights to load.
            src_subject_word (str): New source subject word.
            target_subject_word (str): New target subject word.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            weight).to(self.device)
        self.pipe.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.src_subject_word = src_subject_word
        self.target_subject_word = target_subject_word
        self.photoswap_pipe = PhotoswapPipeline(
            self.pipe, steps=50, guidance_scale=7.5, LOW_RESOURCE=False)
