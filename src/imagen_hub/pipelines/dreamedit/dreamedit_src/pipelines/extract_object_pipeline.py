from PIL import Image

from imagen_hub.utils.save_image_helper import pil_to_tensor, tensor_to_pil

from imagen_hub.miscmodels import LangSAM, draw_image
from imagen_hub.miscmodels.blip import BLIP_Model
from ..utils.mask_helper import merge_masks



def get_object_caption(image, class_name, BLIP_Model, langSAM):
    image = image.resize((256, 256), Image.LANCZOS)
    masks, boxes, phrases, logits = langSAM.predict(image, class_name, box_threshold=0.1, text_threshold=0.1)
    mask = merge_masks(masks)
    tensor_img = pil_to_tensor(image)
    expanded_mask = mask.unsqueeze(0).expand_as(tensor_img)
    masked_tensor = tensor_img * expanded_mask
    final_image = tensor_to_pil(masked_tensor)
    caption = BLIP_Model.predict_one_image(final_image)
    return caption


class Object_Caption_Extractor():

    def __init__(self, device="cuda"):
        self.BLIP_Model = BLIP_Model(device)
        self.langSAM = LangSAM()

    def get_object_caption(self, image, class_name: str):
        image = image.resize((256, 256), Image.LANCZOS)
        masks, boxes, phrases, logits = self.langSAM.predict(image, class_name, box_threshold=0.1, text_threshold=0.1)
        mask = merge_masks(masks)
        tensor_img = pil_to_tensor(image)
        expanded_mask = mask.unsqueeze(0).expand_as(tensor_img)
        masked_tensor = tensor_img * expanded_mask
        final_image = tensor_to_pil(masked_tensor)
        caption = self.BLIP_Model.predict_one_image(final_image)
        return caption

    def get_langSAM(self):
        return self.langSAM

    def get_BLIP_Model(self):
        return self.BLIP_Model


if __name__ == "__main__":
    image = Image.open("/home/maxku/Research/diffusion_project/data/ref_images/rc_car/found0.jpg")
    extractor = Object_Caption_Extractor()
    caption = extractor.get_object_caption(image, 'rc_car')
    print(caption)
    assert caption == 'a pink toy car with a fireman on top'
