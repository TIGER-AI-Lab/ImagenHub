import torch
from torchvision import transforms
from imagen_hub.pipelines.wuerstchen import WuerstchenPipeline


def tensor2PIL(image):
    return transforms.ToPILImage()(image).convert('RGB')


class WuerstchenModel():
    """
    A wrapper class for Wuerstchen Model,

    References:
        https://huggingface.co/docs/diffusers/v0.23.0/en/api/pipelines/wuerstchen#w%C3%BCrstchen
    """

    def __init__(self, device="cuda"):
        """
        Initializes the Wuerstchen Model

        Args:
            device (str): The device for running the pipeline ("cuda" or "cpu"). Default is "cuda".
        """
        self.pipe = WuerstchenPipeline(device=device)

    def infer_one_image(self, caption, negative_caption, seed=42, height=1024, width=1024):
        """
        Generate a image using Wuerstchen Model

        Args:
            caption: The text provided for the generation
            negative_caption: The negative text provided for the generation
            seed (int, optional): The seed for random generator. Default is 42.
            height: height of the image
            width: width of the image

        Returns:
            PIL.Image.Image: The generated image.

        """
        torch.manual_seed(seed)
        batch_size = 1
        latent_height = 128 * (height // 128) // (1024 // 24)
        latent_width = 128 * (width // 128) // (1024 // 24)
        prior_features_shape = (batch_size, 16, latent_height, latent_width)
        prior_inference_steps = {2 / 3: 20, 0.0: 10}
        effnet_features_shape = (batch_size, 16, 12, 12)
        prior_cfg = 6
        prior_sampler = "ddpm"
        generator_steps = 12
        generator_cfg = None
        generator_sampler = "ddpm"

        clip_text_embeddings, clip_text_embeddings_uncond = self.pipe.embed_clip(caption, negative_caption)
        effnet_embeddings_uncond = torch.zeros(effnet_features_shape).to(self.pipe.device)
        generator_latent_shape = (batch_size, 4, int(latent_height * (256 / 24)), int(latent_width * (256 / 24)))
        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            sampled = None
            for t_end, steps in prior_inference_steps.items():
                sampled = self.pipe.diffuzz.sample(self.pipe.model, {'c': clip_text_embeddings}, x_init=sampled,
                                                   unconditional_inputs={"c": clip_text_embeddings_uncond},
                                                   shape=prior_features_shape,
                                                   timesteps=steps, cfg=prior_cfg, sampler=prior_sampler,
                                                   t_start=t_start, t_end=t_end)[-1]
                t_start = t_end
            sampled = sampled.mul(42).sub(1)

            clip_text_embeddings, clip_text_embeddings_uncond = self.pipe.embed_clip(caption, negative_caption,
                                                                                     batch_size)
            sampled_images_original = \
                self.pipe.diffuzz.sample(self.pipe.generator, {'effnet': sampled, 'clip': clip_text_embeddings},
                                         generator_latent_shape, t_start=1.0, t_end=0.00,
                                         timesteps=generator_steps, cfg=generator_cfg,
                                         sampler=generator_sampler,
                                         unconditional_inputs={
                                             'effnet': effnet_embeddings_uncond,
                                             'clip': clip_text_embeddings_uncond,
                                         })[-1]

        image = self.pipe.decode(sampled_images_original)
        image = image.reshape(3, height, width)
        img = tensor2PIL(image)
        return img
