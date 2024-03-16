import torch
import PIL


class ControlNet():
    """
    ControlNet class for image processing.
    Provides an interface for different tasks like control canny, control depth, and more.

    By default, uses the weights provided by `lllyasviel` from Huggingface model hub.
    More details available at:
    - https://huggingface.co/lllyasviel/ControlNet/tree/main/models
    - https://github.com/lllyasviel/ControlNet-v1-1-nightly
    """
    def __init__(self, device="cuda", use_nightly_weights=True):
        """
        Initialize ControlNet with specified device and weights preference (nightly or stable).

        Args:
            device (str): Device to load the model. Default is "cuda".
            use_nightly_weights (bool): If true, uses the nightly weights. Default is True.
        """
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        from diffusers import UniPCMultistepScheduler

        controlnet_weights = {
            "control_canny": "lllyasviel/sd-controlnet-canny",
            "control_depth": "lllyasviel/sd-controlnet-depth",
            "control_openpose": "lllyasviel/sd-controlnet-openpose",
            "control_grayscale": "lllyasviel/sd-controlnet-depth",
            "control_hed": "lllyasviel/sd-controlnet-hed",
        }
        controlnet_nightly_weights = {
            "control_canny": "lllyasviel/control_v11p_sd15_canny",
            "control_depth": "lllyasviel/control_v11f1p_sd15_depth",
            "control_openpose": "lllyasviel/control_v11p_sd15_openpose",
            "control_grayscale": "lllyasviel/control_v11f1p_sd15_depth",
            "control_hed": "lllyasviel/sd-controlnet-hed",
        }

        self.task2weight = controlnet_nightly_weights if use_nightly_weights else controlnet_weights
        self.pipe_dict = {}
        for task, weight in self.task2weight.items():
            controlnet = ControlNetModel.from_pretrained(weight, torch_dtype=torch.float16)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.to(device)
            self.pipe_dict[task] = self.pipe


    def infer_one_image(self, src_image: PIL.Image.Image = None, prompt: str = None, task: str = "control_canny", seed: int = 42):
        """
        Inference method for ControlNet tasks.

        Args:
            src_image (PIL.Image.Image, optional): Source image in RGB format, pre-processed. Default is None.
            prompt (str, optional): Text prompt for the model. Default is None.
            task (str, optional): Task type for ControlNet. Defaults to "control_canny".
            seed (int, optional): Seed for randomness. Default is 42.

        Returns:
            PIL.Image.Image: Processed image.
        """
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)
        # configs from https://huggingface.co/blog/controlnet

        images = self.pipe_dict[task](
            prompt,
            src_image,
            num_inference_steps=50,
            generator=generator,
        ).images

        return images[0]
