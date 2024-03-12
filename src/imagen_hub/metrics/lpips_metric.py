from torchvision import transforms
from PIL import Image

class MetricLPIPS():
    """
    A wrapper for the LPIPS metric.
    Reference: https://github.com/richzhang/PerceptualSimilarity#a-basic-usage

    Provides methods to evaluate perceptual similarity between two images using 
    Learned Perceptual Image Patch Similarity.
    """
    def __init__(self, device="cuda", net_type: str = 'alex') -> None:
        """
        Initializes the MetricLPIPS.

        Args:
            device (str, optional): Device for the model. Defaults to 'cuda'.
            net_type (str, optional): The architecture type for LPIPS (either 'alex' or 'vgg'). Defaults to 'alex'.
        """
        import lpips
        self.device = device
        self.model = lpips.LPIPS(net=net_type).to(self.device)

    def evaluate(self, real_image: Image.Image, generated_image: Image.Image):
        """
        Computes the LPIPS distance between a real image and a generated image.

        Args:
            real_image (Image.Image): The reference image.
            generated_image (Image.Image): The generated image to compare with the real image.
        
        Returns:
            float: LPIPS distance between the two images.
        """
        return evaluate_lpips_score(real_image, generated_image, self.model, self.device)


def get_lpips_alex_model():
    """
    Loads and returns the LPIPS model with AlexNet backbone.

    Returns:
        lpips.LPIPS: The LPIPS model with AlexNet backbone.
    """
    import lpips
    return lpips.LPIPS(net='alex')  # best forward scores


def get_lpips_vgg_model():
    """
    Loads and returns the LPIPS model with VGG backbone.

    Returns:
        lpips.LPIPS: The LPIPS model with VGG backbone.
    """
    # closer to "traditional" perceptual loss, when used for optimization
    import lpips
    return lpips.LPIPS(net='vgg')


def evaluate_lpips_score(real_image, generated_image, lpips_model, device):
    """
    Computes the LPIPS distance between a real image and a generated image using a provided model.

    Args:
        real_image (Image.Image): The reference image.
        generated_image (Image.Image): The generated image to compare.
        lpips_model (lpips.LPIPS): The LPIPS model to use for the evaluation.
        device (str): The device to use for computation.
    
    Returns:
        float: LPIPS distance between the two images.
    """
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
