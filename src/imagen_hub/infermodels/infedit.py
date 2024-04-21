import torch
import PIL

from diffusers import StableDiffusionPipeline, LCMScheduler

class InfEdit():
    """
    A class for InfEdit.
    
    References: https://github.com/sled-group/InfEdit
    """

    def __init__(self, device="cuda", weight="SimianLuo/LCM_Dreamshaper_v7", src_subject_word=None, target_subject_word=None):
        """
        Initialize the InfEdit class.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            weight (str, optional): Pretrained model weight and schuler name. Defaults to "SimianLuo/LCM_Dreamshaper_v7".
            src_subject_word (str, optional): Source subject word. Defaults to None.
            target_subject_word (str, optional): Target subject word. Defaults to None.
        """
        from imagen_hub.pipelines.infedit.pipeline_infedit import InfEditPipeline
        from imagen_hub.pipelines.infedit.pipeline_ddcm import EditPipeline

        self.device = device
        self.torch_dtype= torch.float16 if  device =="cuda" else torch.float32
        scheduler = LCMScheduler.from_config(weight,subfolder="scheduler")
        self.pipe = EditPipeline.from_pretrained(weight, scheduler=scheduler, torch_dtype=self.torch_dtype,safety_checker=None).to(self.device)
        try:
            self.pipe.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        self.ptp_pipe = InfEditPipeline(
            self.pipe, steps=15, guidance_scale_s=1, guidance_scale_t=2, LOW_RESOURCE=False,device = device,torch_dtype =self.torch_dtype)

        self.src_subject_word = src_subject_word
        self.target_subject_word = target_subject_word

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None,height = 512,width=512, seed: int = 42):
        """
        Perform inference on a source image based on provided prompts.

        Args:
            src_image (PIL.Image): Source image.
            src_prompt (str, optional): Description or caption of the source image.
            target_prompt (str, optional): Desired description or caption for the output image.
            instruct_prompt (str, optional): Instruction prompt. Not utilized in this implementation.
            
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            PIL.Image: Transformed image based on the provided prompts.
        """
        torch.manual_seed(seed)
        ratio = min(height / src_image.height, width / src_image.width)
        src_image = src_image.resize((int(src_image.width * ratio), int(src_image.height * ratio)))
        
        controller = self.ptp_pipe.get_controller(src_prompt, target_prompt, source_subject_word=self.src_subject_word, target_subject_word=self.target_subject_word,  denoise=False , strength=0.7,
               cross_replace_steps=0.7, self_replace_steps=0.6, thresh_e=0.3, thresh_m=0.3)
        image = self.ptp_pipe.generate_image(src_image,src_prompt, target_prompt,
                                             controller)
        return image
