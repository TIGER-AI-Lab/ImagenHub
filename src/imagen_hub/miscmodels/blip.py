import torch
import PIL

from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIP_Model():
    """
    A class to represent the BLIP (Baseline Image Processing) model.
    """
    def __init__(self, device="cuda", weight = "Salesforce/blip-image-captioning-base"):
        """
        Initialize a BLIP_Model object with the specified weight and device.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
            weight (str, optional): The pretrained weights to use for the model. 
                                    Defaults to "Salesforce/blip-image-captioning-base".
        """
        self.processor = BlipProcessor.from_pretrained(weight)
        self.model = BlipForConditionalGeneration.from_pretrained(weight)
        self.device = device
        self.model.to(self.device)

    def predict_one_image(self, image):
        """
        Generate a prediction text for a given image.
        
        Args:
            image (PIL.Image.Image): Image for which the prediction text is to be generated.
            
        Returns:
            str: The generated prediction text.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

if __name__ == "__main__":

    import requests

    BLIP_Model = BLIP_Model()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    generated_text = BLIP_Model.predict_one_image(image)
    #print(generated_text)
    assert generated_text == "two cats sleeping on a couch"
