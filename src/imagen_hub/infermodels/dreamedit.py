import torch
import PIL

class DreamEdit():
    """
    DreamEdit provides an image editing functionality using a custom implementation of the DreamEditPipeline.

    Currently, only the "replace" task is implemented.
    """

    def __init__(self, device="cuda") -> None:
        """
        Attributes:
            pipe (DreamEditPipeline): The main pipeline used for image editing.

        Args:
            device (str, optional): Device to load the pipeline on. Default is "cuda".
        """
        from imagen_hub.pipelines.dreamedit.pipeline_dreamedit import DreamEditPipeline
        self.pipe = DreamEditPipeline(device=device)
        self.pipe.set_default_config("replace") # Only implemented replacement task for now

    def infer_one_image(self,
                        src_image: PIL.Image.Image,
                        src_prompt: str,
                        target_prompt: str,
                        seed: int = 42):
        """
        Generate an edited image based on source image and given prompts.

        Args:
            src_image (PIL.Image.Image): Source image to be edited.
            src_prompt (str): Source image caption.
            target_prompt (str): Target image caption.
            seed (int, optional): Random seed. Default is 42.

        Returns:
            PIL.Image.Image: Edited image.
        """
        self.pipe.set_prompts(src_prompt=src_prompt, dst_prompt=target_prompt)
        self.pipe.set_ddim_config(ddim_steps=40, scale=5.5, noise_step=0, iteration_number=5)
        self.pipe.set_seed(seed=seed)

        output = self.pipe.infer(src_image=src_image, obj_image=None)
        return output

    def load_new_subject_weight(self, weight, src_subject_word, target_subject_word):
        """
        Load a custom checkpoint file for the pipeline instead of standard diffusers weight.
        Note: This method will need further implementation. TODO let it support diffusers weight

        Args:
            weight (str): Path to the checkpoint file.
            src_subject_word (str): Source subject word for editing.
            target_subject_word (str): Target subject word for editing.
        """
        self.pipe.set_sd_model(ckpt_path=weight)
        self.pipe.set_subject(src_subject_word, target_subject_word)
