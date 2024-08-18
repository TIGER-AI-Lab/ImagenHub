import torch
import os
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

class Kolors():
    """
    Kolors is a large-scale text-to-image generation model based on latent diffusion, developed by the Kuaishou Kolors team.
    Reference: https://huggingface.co/Kwai-Kolors/Kolors#kolors-effective-training-of-diffusion-model-for-photorealistic-text-to-image-synthesis
    """
    def __init__(self, device="cuda", weight="Kwai-Kolors/Kolors"):
        """
        Attributes:
            pipe (StableDiffusionXLPipeline): The underlying image generation pipeline object.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for image generation. 
        """
        from imagen_hub import MODEL_PATH
        from huggingface_hub import snapshot_download, hf_hub_download
        from imagen_hub.pipelines.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
        from imagen_hub.pipelines.kolors.models.modeling_chatglm import ChatGLMModel
        from imagen_hub.pipelines.kolors.models.tokenization_chatglm import ChatGLMTokenizer
        ckpt_dir = os.path.join(MODEL_PATH, "Kolors")
        ckpt_dir = snapshot_download(weight,
                                       local_dir=ckpt_dir)
        text_encoder = ChatGLMModel.from_pretrained(
            os.path.join(ckpt_dir, "text_encoder"),
            torch_dtype=torch.float16).half()
        tokenizer = ChatGLMTokenizer.from_pretrained(os.path.join(ckpt_dir, "text_encoder"))
        vae = AutoencoderKL.from_pretrained(os.path.join(ckpt_dir, "vae"), revision=None).half()
        scheduler = EulerDiscreteScheduler.from_pretrained(os.path.join(ckpt_dir, "scheduler"))
        unet = UNet2DConditionModel.from_pretrained(os.path.join(ckpt_dir, "unet"), revision=None).half()
        self.pipe = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                force_zeros_for_empty_prompt=False)
        self.device = device

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.to(self.device)
        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=generator).images[0]

        return image
