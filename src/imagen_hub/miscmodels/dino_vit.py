import torch
from torchvision import transforms
from PIL import Image

class VITs16():
    def __init__(self, device="cuda"):
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        self.model.eval()
        self.device = device
        
    def get_transform(self):
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return val_transform

    def get_embeddings(self, tensor_image):
        output = self.model(tensor_image.to(self.device))
        return output

    def get_embeddings_intermediate(self, tensor_image, n_last_block=4):
        """
        We use `n_last_block=4` when evaluating ViT-Small
        """
        intermediate_output = self.model.get_intermediate_layers(tensor_image, n=n_last_block)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        return output

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import requests    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = image.resize((512, 512))
    fidelity = VITs16(device)
    preprocess = fidelity.get_transform()
    image = preprocess(image)
    print(image.shape)
    image = image.unsqueeze(0)
    result = fidelity.get_embeddings(image)
    assert result.shape == torch.Size([1, 384])