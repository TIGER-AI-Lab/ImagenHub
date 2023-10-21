import clip
import torch
import PIL

class CLIP():
    """
    A class to represent the CLIP (Contrastive Language-Image Pre-training) model.
    """
    def __init__(self, device="cuda"):
        """
        Initialize a CLIP object with the specified device.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.device = device

    def get_transform(self):
        """
        Returns the preprocessing transforms.
        
        Returns:
            torchvision.transforms.Compose: Preprocessing transforms.
        """
        return self.preprocess

    def encode_image(self, tensor_image):
        """
        Encode the provided tensor image into a feature tensor.
        
        Args:
            tensor_image (torch.Tensor): Image tensor of shape [B, 3, H, W].
            
        Returns:
            torch.Tensor: Encoded image tensor.
        """
        output = self.model.encode_image(tensor_image.to(self.device))
        return output

    def encode_text(self, prompt):
        """
        Encode the provided text into a feature tensor.
        
        Args:
            prompt (str): The text to encode.
            
        Returns:
            torch.Tensor: Encoded text tensor.
        """
        text = clip.tokenize([prompt], context_length=77, truncate=True).to(self.device)
        text_features = self.model.encode_text(text)
        return text_features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIP(device)

    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = PIL.Image.open(requests.get(url, stream=True).raw).convert('RGB')
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
