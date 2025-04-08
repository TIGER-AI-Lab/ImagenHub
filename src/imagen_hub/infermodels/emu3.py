class Emu3:
    def __init__(self, model_path="BAAI/Emu3-Gen",
                 vq_path="BAAI/Emu3-VisionTokenizer",
                 device="cuda:0",
                 guidance_scale=3.0):
        """
        Attributes:
            pipe (Emu3Pipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "BAAI/Emu3-Gen".
        """
        from imagen_hub.pipelines.emu3 import Emu3Pipeline
        self.pipe = Emu3Pipeline(model_path=model_path, vq_path=vq_path, device=device, guidance_scale = guidance_scale)
        

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