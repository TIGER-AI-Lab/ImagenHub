import clip
from PIL import Image
import torch

class CLIP():
    def __init__(self, device="cuda"):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.device = device
    
    def get_transform(self):
        return self.preprocess
    
    def encode_image(self, tensor_image):
        """
        Take input in size [B, 3, H, W]
        """
        output = self.model.encode_image(tensor_image.to(self.device))
        return output
    
    def encode_text(self, prompt):
        text = clip.tokenize([prompt], context_length=77, truncate=True).to(self.device) 
        text_features = self.model.encode_text(text)
        return text_features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIP(device)
    
    import requests    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = image.resize((512, 512))
    preprocess = clip_model.get_transform()
    image = preprocess(image)
    image = image.unsqueeze(0)
    assert image.shape == torch.Size([1, 3, 224, 224])
    result = clip_model.encode_image(image)
    assert result.shape == torch.Size([1, 512])
    text = "hello world"
    result = clip_model.encode_text(text)
    assert result.shape == torch.Size([1, 512])