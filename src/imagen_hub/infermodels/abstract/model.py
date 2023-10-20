
from abc import ABC, abstractmethod
from PIL import Image


class BaseModel(ABC):
    """
    Abstarct class for BaseModels.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError("Please Implement this method")

class TextToImageGenerationModel(BaseModel):
    """
    Abstarct class for Text-To-Image Generation Models.
    """
    @abstractmethod
    def infer_one_image(self, prompt: str, seed: int = 42, **kwargs):
        """
        param:
            src_image : PIL Image in RGB format
            prompt : prompt
            seed : random seed
            **kwargs : other key arguments
        return:
            output image : PIL image
        """
        raise NotImplementedError("Please Implement this method")


class TextGuidedImageEditingModel(BaseModel):
    """
    Abstarct class for Text-Guided Image Editing Models.
    """
    @abstractmethod
    def infer_one_image(self, src_image: Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42, **kwargs):
        """
        param:
            src_image : PIL Image
            src_prompt : source image caption
            target_prompt : target image caption
            instruct_prompt : instruction prompt
            seed : seed
            **kwargs : other key arguments
        return:
            output image
        """
        raise NotImplementedError("Please Implement this method")


class MaskGuidedImageEditingModel(BaseModel):
    """
    Abstarct class for Mask-Guided Image Editing Models.
    """
    @abstractmethod
    def infer_one_image(self, src_image: Image = None, local_mask_prompt: str = None, mask_image: Image = None, seed: int = 42, **kwargs):
        """
        param:
            src_image : PIL Image
            local_mask_prompt : target image caption
            mask_image : PIL Image mask for inpainting
            seed : random seed
            **kwargs : other key arguments
        return:
            output image : PIL image
        """
        raise NotImplementedError("Please Implement this method")


class ControlGuidedImageGenerationModel(BaseModel):
    """
    Abstarct class for Control-Guided Image Generation Model.
    """
    @abstractmethod
    def infer_one_image(self, src_image: Image = None, prompt: str = None, task: str = "control_canny", seed: int = 42, **kwargs):
        """
        param:
            src_image : pre-processed Image
            prompt :  prompt
            task :  task | control_canny, ...
            seed : random seed
            **kwargs : other key arguments
        return:
            output image
        """
        raise NotImplementedError("Please Implement this method")

class SubjectDrivenImageGenerationModel(BaseModel):
    """
    Abstarct class for Subject-Driven Image Generation Model.
    """
    @abstractmethod
    def infer_one_image(self, **kwargs):
        raise NotImplementedError("Please Implement this method")

class SubjectDrivenImageEditingModel(BaseModel):
    """
    Abstarct class for Subject-Driven Image Editing Model.
    """
    @abstractmethod
    def infer_one_image(self, **kwargs):
        raise NotImplementedError("Please Implement this method")