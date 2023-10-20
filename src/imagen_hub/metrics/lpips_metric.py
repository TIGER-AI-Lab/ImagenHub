import lpips
# https://github.com/richzhang/PerceptualSimilarity#a-basic-usage
from torchvision import transforms
from PIL import Image

"""
Learned Perceptual Image Patch Similarity (LPIPS) metric
"""


class MetricLPIPS():
    def __init__(self, device="cuda", net_type: str = 'alex') -> None:
        self.device = device
        self.model = lpips.LPIPS(net=net_type).to(self.device)

    def evaluate(self, real_image: Image.Image, generated_image: Image.Image):
        """
        return distance
        """
        return evaluate_lpips_score(real_image, generated_image, self.model, self.device)


def get_lpips_alex_model():
    return lpips.LPIPS(net='alex')  # best forward scores


def get_lpips_vgg_model():
    # closer to "traditional" perceptual loss, when used for optimization
    return lpips.LPIPS(net='vgg')


def evaluate_lpips_score(real_image, generated_image, lpips_model, device):
    generated_image = generated_image.resize((512,512)) 
    real_image = real_image.resize((512,512)) 

    transform = transforms.Compose([
        transforms.ToTensor(),
        # Range needs to be [-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    real_img_tensor = transform(real_image).unsqueeze(0).float().to(device)
    generated_img_tensor = transform(
        generated_image).unsqueeze(0).float().to(device)
    distance = lpips_model(real_img_tensor, generated_img_tensor)
    return distance.item()
