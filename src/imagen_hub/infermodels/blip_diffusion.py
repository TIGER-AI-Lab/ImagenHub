import torch
import PIL

class BLIPDiffusion():
    """
    Base class for BLIP Diffusion. This class provides utility methods and loads the BLIP diffusion model.
    """
    def __init__(self, device="cuda", weight="blip_diffusion"):
        """
        Initialize BLIPDiffusion with specified device and weights.

        Args:
            device (str): Device to load the model. Default is "cuda".
            weight (str): Weight name for the diffusion model. Default is "blip_diffusion".
        """
        from lavis.models import load_model_and_preprocess
        self.device = device
        self.pipe, self.vis_preprocess, self.txt_preprocess = load_model_and_preprocess(weight,
                                                                                        "base", device=self.device,
                                                                                        is_eval=True)

    def get_pipe(self):
        """
        Retrieves the model pipeline and preprocessing utilities.

        Returns:
            tuple: pipe, visual preprocess, text preprocess.
        """
        return self.pipe, self.vis_preprocess, self.txt_preprocess

class BLIPDiffusion_Gen(BLIPDiffusion):
    """
    Child class for zero-shot image generation using BLIP Diffusion.

    Reference: https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/generation_zeroshot.ipynb
    """
    def __init__(self, device="cuda", weight="blip_diffusion"):
        """
        Initialize BLIPDiffusion_Gen with specified device and weights.

        Args:
            device (str): Device to load the model. Default is "cuda".
            weight (str): Weight name for the diffusion model. Default is "blip_diffusion".
        """
        super().__init__(device, weight)
        self.device = device
        self.pipe, self.vis_preprocess, self.txt_preprocess = self.get_pipe()

    def infer_one_image(self,
                        cond_image: PIL.Image.Image,
                        text_prompt: str,
                        cond_subject_name: str,
                        target_subject_name: str=None,
                        seed: int = 42):
        """
        Inference method for zero-shot image generation.

        Args:
            cond_image (PIL.Image.Image): Conditioning image in RGB format.
            text_prompt (str): Description of the target image without the subject.
            cond_subject_name (str): Subject present in the cond_image (e.g., dog).
            target_subject_name (str, optional): Subject intended to be generated, similar to cond_subject. Defaults to None.
            seed (int, optional): Seed for randomness. Default is 42.

        Returns:
            PIL.Image.Image: Generated image.
        """
        cond_image = cond_image.convert('RGB') # force it to RGB format
        cond_images = self.vis_preprocess["eval"](cond_image).unsqueeze(0).to(self.device)
        cond_subjects = [self.txt_preprocess["eval"](cond_subject_name)]

        if target_subject_name is None:
            target_subject_name = cond_subject_name

        tgt_subjects = [self.txt_preprocess["eval"](target_subject_name)]
        text_prompt = [self.txt_preprocess["eval"](text_prompt)]

        samples = {
            "cond_images": cond_images,
            "cond_subject": cond_subjects,
            "tgt_subject": tgt_subjects,
            "prompt": text_prompt,
        }

        image = self.pipe.generate(
            samples,
            seed=seed,
            guidance_scale=7.5,
            num_inference_steps=50,
            neg_prompt="", # No negative prompt
            height=512,
            width=512,
        )

        return image[0]

class BLIPDiffusion_Edit(BLIPDiffusion):
    """
    Child class for editing images using BLIP Diffusion.

    Reference: https://github.com/salesforce/LAVIS/blob/main/projects/blip-diffusion/notebooks/editing_real_zeroshot.ipynb
    """
    def __init__(self, device="cuda", weight="blip_diffusion"):
        """
        Initialize BLIPDiffusion_Edit with specified device and weights.

        Args:
            device (str): Device to load the model. Default is "cuda".
            weight (str): Weight name for the diffusion model. Default is "blip_diffusion".
        """
        super().__init__(device, weight)
        self.device = device
        self.pipe, self.vis_preprocess, self.txt_preprocess = self.get_pipe()

    def infer_one_image(self,
                        src_image: PIL.Image.Image,
                        cond_image: PIL.Image.Image,
                        text_prompt: str,
                        src_subject_name: str,
                        cond_subject_name: str,
                        target_subject_name: str = None,
                        seed: int = 42):
        """
        Inference method for editing images.

        Args:
            src_image (PIL.Image.Image): Source image in RGB format.
            cond_image (PIL.Image.Image): Conditioning image in RGB format.
            text_prompt (str): Description of the target image without the subject.
            src_subject_name (str): Subject in the source image.
            cond_subject_name (str): Subject present in the cond_image.
            target_subject_name (str, optional): Subject intended to be generated, similar to cond_subject. Defaults to None.
            seed (int, optional): Seed for randomness. Default is 42.

        Returns:
            PIL.Image.Image: Edited image.
        """
        src_image = src_image.convert('RGB') # force it to RGB format

        cond_image = cond_image.convert('RGB') # force it to RGB format
        cond_images = self.vis_preprocess["eval"](cond_image).unsqueeze(0).to(self.device)

        if target_subject_name is None:
            target_subject_name = cond_subject_name

        src_subject = self.txt_preprocess["eval"](src_subject_name)
        tgt_subject = self.txt_preprocess["eval"](target_subject_name)
        cond_subject = self.txt_preprocess["eval"](cond_subject_name)
        text_prompt = [self.txt_preprocess["eval"](text_prompt)]

        samples = {
            "cond_images": cond_images,
            "cond_subject": cond_subject,
            "src_subject": src_subject,
            "tgt_subject": tgt_subject,
            "prompt": text_prompt,
            "raw_image": src_image,
        }
        output = self.pipe.edit(
            samples,
            seed=seed,
            guidance_scale=7.5,
            num_inference_steps=50,
            num_inversion_steps=50,
            neg_prompt="", # No negative prompt
        )[1] #[1] is the edited


        return output
