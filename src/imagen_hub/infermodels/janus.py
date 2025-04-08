class Janus:
    def __init__(self, device="cuda",weight="deepseek-ai/Janus-1.3B"):
        """
        Attributes:
            pipe (JanusPipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "deepseek-ai/Janus-1.3".
        """
        from imagen_hub.pipelines.janus import JanusPipeline
        self.pipe = JanusPipeline(weight= weight, device=device)
        

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

class JanusPro(Janus):
    def __init__(self, device="cuda"):
        """
        Inherits from Janus but uses the 7B model weights.
        
        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
        """
        super().__init__(device=device, weight="deepseek-ai/Janus-Pro-7B")

class JanusFlow:
    def __init__(self, device="cuda",weight="deepseek-ai/JanusFlow-1.3B"):
        """
        Attributes:
            pipe (JanusPipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "deepseek-ai/Janus-1.3".
        """
        from imagen_hub.pipelines.janus import JanusFlowPipeline
        self.pipe = JanusFlowPipeline(weight= weight, device=device)
        

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