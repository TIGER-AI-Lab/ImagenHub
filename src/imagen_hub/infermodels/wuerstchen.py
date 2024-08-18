import torch

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
        import accelerate
        import peft
        from diffusers import WuerstchenPriorPipeline, WuerstchenDecoderPipeline
        from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
        self.stage_c_timesteps = DEFAULT_STAGE_C_TIMESTEPS
        self.device = device
        self.prior_pipeline = WuerstchenPriorPipeline.from_pretrained("warp-ai/wuerstchen-prior",
                                                                      torch_dtype=dtype,
                                                                      low_cpu_mem_usage=True)
        self.decoder_pipelne = WuerstchenDecoderPipeline.from_pretrained("warp-ai/wuerstchen",
                                                                         torch_dtype=dtype,
                                                                         low_cpu_mem_usage=True)
        
    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Generate a image using Wuerstchen Model

        Args:
            prompt: The text provided for the generation.
            seed (int, optional): The seed for random generator. Default is 42.
        Returns:
            PIL.Image.Image: The generated image.

        """
        self.pipe.to(self.device)
        negative_caption = None
        torch.manual_seed(seed)
        prior_output = self.prior_pipeline(prompt=prompt,
                                           timesteps=self.stage_c_timesteps,
                                           negative_caption=negative_caption,
                                           guidance_scale=4.0,
                                           num_images_per_prompt=1)
        decoder_output = self.decoder_pipelne(image_embeddings=prior_output.image_embeddings,
                                              prompt=prompt,
                                              negative_caption=negative_caption,
                                              guidance_scale=0.0,
                                              output_type='pil')
        return decoder_output.images[0]
