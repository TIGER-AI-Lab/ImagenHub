import torch
from diffusers import DiffusionPipeline

class QwenImage:
    def __init__(self,
                 weight: str = "Qwen/Qwen-Image",
                 device: str = None,
                 torch_dtype: torch.dtype = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            self.torch_dtype = torch_dtype
        self.pipe = DiffusionPipeline.from_pretrained(weight, torch_dtype=self.torch_dtype)
        self.pipe = self.pipe.to(self.device)
        self.positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.",
            "zh": "超清，4K，电影级构图"
        }

        self.aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
    
    def infer_one_image(self,
                       prompt: str,
                       negative_prompt: str = "",
                       language: str = "en",
                       aspect_ratio: str = "16:9",
                       num_inference_steps: int = 50,
                       true_cfg_scale: float = 4.0,
                       seed: int = 42):
        if aspect_ratio not in self.aspect_ratios:
            raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

        width, height = self.aspect_ratios[aspect_ratio]
        full_prompt = prompt + " " + self.positive_magic.get(language, "")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator
        )

        return result.images[0]
