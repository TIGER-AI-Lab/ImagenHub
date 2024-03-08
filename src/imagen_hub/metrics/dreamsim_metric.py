from torchvision import transforms
from PIL import Image


class MetricDreamSim():
    """
    A wrapper for the DreamSim metric.

    DreamSim is a perceptual similarity metric designed to capture human visual similarity.
    Reference: https://github.com/ssundaram21/dreamsim/blob/main/dreamsim/model.py
    """
    def __init__(self, device="cuda") -> None:
        """
        Initializes the DreamSim metric wrapper.

        Args:
            device (str, optional): The device to load the model on. Defaults to "cuda".
        """
        self.model, self.preprocess = get_dreamsim_model()
        self.device = device

    def evaluate(self, real_image: Image.Image, generated_image: Image.Image):
        """
        Computes the DreamSim distance between a real image and a generated image.

        Args:
            real_image (Image.Image): The reference or real image.
            generated_image (Image.Image): The generated or output image.

        Returns:
            float: The DreamSim distance between the two images.
        """
        #return evaluate_dreamsim_score(real_image, generated_image, self.model, self.device)
        img1 = self.preprocess(real_image).to("cuda")
        img2 = self.preprocess(generated_image).to("cuda")
        distance = self.model(img1, img2).float().item()
        return distance

def get_dreamsim_model():
    """
    Loads and returns the pre-trained DreamSim model and its preprocessing function.

    Returns:
        tuple: A tuple containing the DreamSim model and its preprocessing function.
    """
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True)
    return model, preprocess

def evaluate_dreamsim_score(real_image, generated_image, dreamsim_model, device):
    """
    Computes the DreamSim distance for a pair of images using a given model.

    Args:
        real_image (Image.Image): The reference or real image.
        generated_image (Image.Image): The generated or output image.
        dreamsim_model (torch.nn.Module): The DreamSim model to use for the evaluation.
        device (str): The device to which the model is loaded ("cuda" or "cpu").

    Returns:
        float: The DreamSim distance between the two images.
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
