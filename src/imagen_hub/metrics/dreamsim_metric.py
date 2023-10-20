from dreamsim import dreamsim
from torchvision import transforms
from PIL import Image


class MetricDreamSim():
    """
    DreamSim metrics
    # https://github.com/ssundaram21/dreamsim/blob/main/dreamsim/model.py
    """
    def __init__(self, device="cuda") -> None:
        self.model, self.preprocess = get_dreamsim_model()
        self.device = device

    def evaluate(self, real_image: Image.Image, generated_image: Image.Image):
        """
        return distance
        """
        #return evaluate_dreamsim_score(real_image, generated_image, self.model, self.device)
        img1 = self.preprocess(real_image).to("cuda")
        img2 = self.preprocess(generated_image).to("cuda")
        distance = self.model(img1, img2).float().item()
        return distance

def get_dreamsim_model():
    model, preprocess = dreamsim(pretrained=True)
    return model, preprocess

def evaluate_dreamsim_score(real_image, generated_image, dreamsim_model, device):
    """
    real_image : reference image
    generated_image : generated image
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    real_img_tensor = transform(real_image).unsqueeze(0).float().to(device)
    generated_img_tensor = transform(generated_image).unsqueeze(0).float().to(device)
    distance = dreamsim_model(real_img_tensor, generated_img_tensor).float()
    return distance
