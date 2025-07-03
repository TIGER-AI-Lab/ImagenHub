class BagelGeneration:
    def __init__(self, weight="ByteDance-Seed/BAGEL-7B-MoT"):
        """
        Attributes:
            pipe (BagelPipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "ByteDance-Seed/BAGEL-7B-MoT".
        """
        from imagen_hub.pipelines.bagel import BagelPipeline
        self.pipe = BagelPipeline(weight= weight)
        

    def infer_one_image(self, prompt: str = None, input_images= [], think = False, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.set_seed(seed)
        image = self.pipe.generate_image(prompt=prompt,input_images=input_images, think=think)
        return image
    
class BagelEdit:
    def __init__(self, weight="ByteDance-Seed/BAGEL-7B-MoT"):
        """
        Attributes:
            pipe (BagelPipeline): The main pipeline used for image generation.

        Args:
            device (str, optional): The device on which the pipeline should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for stable-cascade. Default is "ByteDance-Seed/BAGEL-7B-MoT".
        """
        from imagen_hub.pipelines.bagel import BagelPipeline
        self.pipe = BagelPipeline(weight= weight)
        

    def infer_one_image(self, prompt: str = None, input_images= [], think = False, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """
        self.pipe.set_seed(seed)
        image = self.pipe.edit_image(prompt=prompt,input_images=input_images, think=think)
        return image