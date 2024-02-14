import torch
from diffusers import WuerstchenPriorPipeline, WuerstchenDecoderPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
import accelerate
import peft


class Wuerstchen():
    """
    A wrapper class for Wuerstchen Model,

    References:
        https://huggingface.co/docs/diffusers/v0.23.0/en/api/pipelines/wuerstchen#w%C3%BCrstchen
    """

    def __init__(self, device="cuda", dtype=torch.float16):
        """
        Initializes the Wuerstchen Model
        Using the Pipeline provided in diffusers

        Args:
            device (str): The device for running the pipeline ("cuda" or "cpu"). Default is "cuda".
            dtype (torch.dtype): type that we use after loading the pretrained model.
        """
        self.prior_pipeline = WuerstchenPriorPipeline.from_pretrained("warp-ai/wuerstchen-prior",
                                                                      torch_dtype=dtype,
                                                                      low_cpu_mem_usage=True).to(device)
        self.decoder_pipelne = WuerstchenDecoderPipeline.from_pretrained("warp-ai/wuerstchen",
                                                                         torch_dtype=dtype,
                                                                         low_cpu_mem_usage=True).to(device)

    def infer_one_image(self, caption, negative_caption, seed=42, height=1024, width=1024):
        """
        Generate a image using Wuerstchen Model

        Args:
            caption: The text provided for the generation
            negative_caption: The negative text provided for the generation
            seed (int, optional): The seed for random generator. Default is 42.
            height: height of the image
            width: width of the image

        Returns:
            PIL.Image.Image: The generated image.

        """
        torch.manual_seed(seed)
        prior_output = self.prior_pipeline(prompt=caption,
                                           height=height,
                                           width=width,
                                           timesteps=DEFAULT_STAGE_C_TIMESTEPS,
                                           negative_caption=negative_caption,
                                           guidance_scale=4.0,
                                           num_images_per_prompt=1)
        decoder_output = self.decoder_pipelne(image_embeddings=prior_output.image_embeddings,
                                              prompt=caption,
                                              negative_caption=negative_caption,
                                              guidance_scale=0.0,
                                              output_type='pil')
        return decoder_output.images[0]
