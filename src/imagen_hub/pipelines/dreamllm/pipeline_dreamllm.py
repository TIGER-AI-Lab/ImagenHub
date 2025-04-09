import os
import torch
from transformers import LlamaTokenizer
from huggingface_hub import snapshot_download

from .dreamllm_src.omni.models.dreamllm.configuration_dreamllm import DreamLLMConfig
from .dreamllm_src.omni.models.dreamllm.modeling_dreamllm import DreamLLMForCausalMLM 
from .dreamllm_src.omni.utils.profiler import FunctionProfiler
import numpy as np
import json
import sys
sys.path.append("../../../../src")
import os

import cv2
import numpy as np
import PIL.Image
import torch
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer

class DreamLLMPipeline:
    def __init__(
        self,
        weight="RunpeiDong/dreamllm-7b-chat-aesthetic-v1.0",
        local_files_only=False,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        torch.cuda.empty_cache()
        self.device = device
        self.local_files_only = local_files_only
        self.model_checkpoint = os.path.join(os.path.dirname(__file__), "../../../../checkpoint/dreamllm")
        self.model_checkpoint = os.path.abspath(self.model_checkpoint)
        print(self.model_checkpoint)
        if not os.path.exists(self.model_checkpoint):
            os.makedirs(self.model_checkpoint, exist_ok=True)

        # Download model snapshot if needed
        snapshot_download(weight, local_dir=self.model_checkpoint)

        config_path = os.path.join(self.model_checkpoint, "config.json")  
        old_prefix = "omni."
        new_prefix = "imagen_hub.pipelines.dreamllm.dreamllm_src.omni."
        self.patch_config_targets(config_path, old_prefix, new_prefix)

        # Load tokenizer, config, and model
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_checkpoint, local_files_only=local_files_only,padding_side="right",)
        config = DreamLLMConfig.from_pretrained(self.model_checkpoint, local_files_only=local_files_only)
        self.model = DreamLLMForCausalMLM.from_pretrained(
                self.model_checkpoint,
                tokenizer=self.tokenizer,
                config=config,
                local_files_only=True,       # <-- avoid touching the internet
                cache_dir=self.model_checkpoint,
        ).to(dtype=dtype).to(device)
        self.model = torch.compile(self.model)
    
    def patch_config_targets(self,config_path, old_prefix, new_prefix):
        """
        Patches the JSON config file at config_path by replacing any _target_
        value that starts with old_prefix with new_prefix.
        
        Args:
            config_path (str): The file path to the JSON config file.
            old_prefix (str): The prefix in the _target_ value to replace.
            new_prefix (str): The new prefix to substitute.
        """
        # Load the config
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Recursive function to patch _target_ keys
        def patch_targets(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "_target_" and isinstance(v, str) and v.startswith(old_prefix):
                        obj[k] = v.replace(old_prefix, new_prefix, 1)
                    else:
                        patch_targets(v)
            elif isinstance(obj, list):
                for item in obj:
                    patch_targets(item)

        patch_targets(config)

        # Save the modified config back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Config patched successfully at {config_path}")

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def generate_image(
        self,
        prompt,
        positive_prompt="(best quality,extremely detailed)",
        negative_prompt="ugly,duplicate,morbid,mutilated,tranny,mutated hands,poorly drawn hands,blurry,bad anatomy,bad proportions,extra limbs,cloned face,disfigured,missing arms,extra legs,mutated hands,fused fingers,too many fingers,unclear eyes,lowers,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry,bad feet,gettyimages",
        guidance_scale=3.5,
        num_inference_steps=150
    ):
        torch.cuda.empty_cache()
        # Compose prompt
        full_prompt = f"{positive_prompt} {prompt}" if positive_prompt else prompt

        # If negative_prompt is a string, wrap it as a list
        if negative_prompt and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # Generate image
        images = self.model.stable_diffusion_pipeline(
            tokenizer=self.tokenizer,
            prompt=[full_prompt],
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        torch.cuda.empty_cache()
        return images[0]
