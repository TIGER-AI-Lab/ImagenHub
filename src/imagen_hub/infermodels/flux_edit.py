import torch
import PIL

class FluxEdit:
    """
    A class for Flux Edit.
    
    References: https://github.com/sayakpaul/flux-image-editing
    """

    def __init__(self, device="cuda", weight="sayakpaul/FLUX.1-dev-edit-v0"):
        """
        Initialize the FluxEdit class.
        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            weight (str, optional): Pretrained model weight name. Defaults to "sayakpaul/FLUX.1-dev-edit-v0".
        """
        from diffusers import FluxControlPipeline, FluxTransformer2DModel

        self.device = device
        self.torch_dtype = torch.bfloat16 if  device =="cuda" else torch.float32

        transformer = FluxTransformer2DModel.from_pretrained(weight, torch_dtype=self.torch_dtype)
        self.pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=self.torch_dtype
        ).to(self.device)
        
    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None,height = 512,width=512, seed: int = 42):
        """
        Perform inference on a source image based on provided prompt.
        Args:
            src_image (PIL.Image): Source image.
            instruct_prompt (str, optional): Instruction prompt.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        Returns:
            PIL.Image: Transformed image based on the provided prompts.
        """
        torch.manual_seed(seed)
        ratio = min(height / src_image.height, width / src_image.width)
        src_image = src_image.resize((int(src_image.width * ratio), int(src_image.height * ratio)))

        image = self.pipe(
            control_image=src_image,
            prompt=instruct_prompt,
            guidance_scale=30., # change this as needed.
            num_inference_steps=50, # change this as needed.
            max_sequence_length=512,
            height=src_image.height,
            width=src_image.width,
            generator=torch.manual_seed(seed)
        ).images[0]
        return image 