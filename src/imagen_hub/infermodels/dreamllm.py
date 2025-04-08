class DreamLLM:
    def __init__(self, device="cuda",weight="RunpeiDong/dreamllm-7b-chat-aesthetic-v1.0"):
        """
        Attributes:
            pipe (DreamLLMPipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "RunpeiDong/dreamllm-7b-chat-aesthetic-v1.0".
        """
        from imagen_hub.pipelines.dreamllm import DreamLLMPipeline
        self.pipe = DreamLLMPipeline(weight= weight, device=device)
        

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.set_seed(seed)
        image = self.pipe.generate_image(prompt=prompt)
        return image