import torch
from torchvision import transforms
import PIL

class VITs16():
    """
    A class to represent the Vision Transformer Small 16 (ViT-S16) model.
    """
    def __init__(self, device="cuda"):
        """
        Initialize a VITs16 object with the specified device.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        self.model.eval()
        self.device = device

    def get_transform(self):
        """
        Returns the preprocessing transforms for the images.
        
        Returns:
            torchvision.transforms.Compose: Preprocessing transforms.
        """
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return val_transform

    def get_embeddings(self, tensor_image):
        """
        Get the final embeddings of the image using the ViT-S16 model.
        
        Args:
            tensor_image (torch.Tensor): Image tensor of shape [B, 3, H, W].
            
        Returns:
            torch.Tensor: Final embeddings tensor.
        """
        output = self.model(tensor_image.to(self.device))
        return output

    def get_embeddings_intermediate(self, tensor_image, n_last_block=4):
        """
        Get the intermediate embeddings of the image using the ViT-S16 model.
        
        Args:
            tensor_image (torch.Tensor): Image tensor of shape [B, 3, H, W].
            n_last_block (int, optional): Number of last blocks to consider for intermediate embeddings. 
                                          Defaults to 4.
            
        Returns:
            torch.Tensor: Intermediate embeddings tensor.
        """
        intermediate_output = self.model.get_intermediate_layers(tensor_image, n=n_last_block)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        return output

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = PIL.Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = image.resize((512, 512))
    fidelity = VITs16(device)
    preprocess = fidelity.get_transform()
    image = preprocess(image)
    print(image.shape)
    image = image.unsqueeze(0)
    result = fidelity.get_embeddings(image)
    assert result.shape == torch.Size([1, 384])
