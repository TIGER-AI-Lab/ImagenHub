import torch
import numpy as np

def latent2image(vae, latents):
    """
    Convert latents to images using a VAE (Variational Autoencoder) decoder.
    Copied from Prompt2Prompt latent2image function
    source: https://github.com/google/prompt-to-prompt/blob/8186230a8ece64c2953271659aa858ec6d02a5cf/ptp_utils.py#L78

    Args:
        vae: The Variational Autoencoder model with a decode method.
        latents (tensor): Latent representations to be decoded into images.

    Returns:
        ndarray: Decoded images in uint8 format.
    """
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def init_latent(latent, model, height: int, width: int, generator, batch_size: int):
    """
    Initialize or expand a given latent tensor.

    If the latent tensor is not provided (None), it initializes a random tensor using the specified dimensions.
    Otherwise, it expands the provided latent tensor to the desired batch size.

    Args:
        latent (tensor, optional): Initial latent tensor or None.
        model: Model with an 'unet' attribute, used to determine the number of in_channels.
        height (int): Height of the images to be generated.
        width (int): Width of the images to be generated.
        generator: Torch generator for random number generation.
        batch_size (int): Desired batch size for the latents.

    Returns:
        tuple: A tuple containing the original latent and the expanded latents tensor.
    """
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents
