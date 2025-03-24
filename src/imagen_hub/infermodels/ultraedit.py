
import torch
import PIL


class UltraEdit():
    """
    This implementation only support Free-form image editing.
    Reference: https://ultra-editing.github.io/
    """
    def __init__(self, device="cuda", weight="BleachNick/SD3_UltraEdit_freeform"):
        """
        Attributes:
            pipe (StableDiffusion3InstructPix2PixPipeline): The SD3InstructPix2Pix pipeline for image transformation.

        Args:
            device (str, optional): Device on which the pipeline runs. Defaults to "cuda".
            weight (str, optional): Pretrained weights for the model.
        """
        from imagen_hub.pipelines.ultraedit.pipeline_stable_diffusion_3_instructpix2pix import StableDiffusion3InstructPix2PixPipeline
        self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.device = device


    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42):
        """
        Modifies the source image based on the provided instruction prompt.

        Args:
            src_image (PIL.Image.Image): Source image in RGB format.
            instruct_prompt (str): Caption for editing the image.
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)
        image = self.pipe(instruct_prompt, 
                          image=src_image,
                          mask_img=None,
                          num_inference_steps=50,
                          image_guidance_scale=1.5,
                          guidance_scale=7.5,
                          generator=generator
                          ).images[0]
        return image