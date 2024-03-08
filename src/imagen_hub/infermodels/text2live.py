import torch
import PIL

class Text2Live():
    """
    Text2Live with default hyperparameters in the original code
    Reference: https://github.com/omerbt/Text2LIVE
    """
    def __init__(self, device="cuda"):
        from imagen_hub.pipelines.text2live import Text2LivePipeline
        
        self.pipe = Text2LivePipeline(device=device)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, screen_text: str =None, epoch=50, seed=42):
        """
        Edits an image based on the given text instructions.

        Args:
            src_image (PIL.Image.Image, optional): The original image to be edited in RGB format.
            screen_text (str, optional): The text describing the edit layer (local object).
            target_prompt (str, optional): The text describing the full edited image (target image).
            src_prompt (str, optional): The text describing the input image (input image).
            epoch (int, optional): The number of epochs to train the model. Default is 50.
            seed (int, optional): The random seed for the model. Default is 42.

        Returns:
            PIL.Image.Image: The edited image.

        For more detailed parameter explanations, refer to:
        https://github.com/omerbt/Text2LIVE/blob/main/configs/image_example_configs/golden_horse.yaml
        """
        src_image = src_image.convert('RGB') # force it to RGB format
        comp_text = target_prompt
        src_text = src_prompt
        if screen_text is None:
            screen_text = target_prompt
        self.pipe.set_seed(seed)
        image_dicts = self.pipe(
            src_image,
            screen_text,
            comp_text,
            src_text,
            epoch=epoch
        )
        return image_dicts['composite']
