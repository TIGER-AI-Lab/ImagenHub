import os
import torch
import subprocess
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer , AutoConfig
from .hart_src.hart.utils import  encode_prompts, llm_system_prompt
from huggingface_hub import snapshot_download
from .hart_src.hart.modules.models.transformer import HARTForT2I

class HartPipeline:
    def __init__(
        self,
        weight="mit-han-lab/hart-0.7b-1024px",
        text_model_path="Qwen2-VL-1.5B-Instruct",
        use_ema=True,
        max_token_length=300,
        use_llm_system_prompt=True,
        cfg=4.5,
        more_smooth=True,
        device="cuda"
    ):
        self.device = device
        self.use_ema = use_ema
        self.max_token_length = max_token_length
        self.use_llm_system_prompt = use_llm_system_prompt
        self.cfg = cfg
        self.more_smooth = more_smooth
        model_path = os.path.join(os.path.dirname(__file__), "../../../../checkpoint/hart")
        model_url = weight
        #auto_clone(model_url, model_path)
        snapshot_download(model_url, local_dir=model_path)
        # Load main model
        model_path = os.path.join(model_path, "llm")
        config = AutoConfig.from_pretrained(model_path)
        config.device = self.device 
        # Load HART model manually
        self.model = HARTForT2I(config=config)
        model_ckpt = os.path.join(model_path, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        


        # Load EMA model if enabled
        self.ema_model = None
        if use_ema:
            config = AutoConfig.from_pretrained(model_path)
            config.device = self.device  # Optional, for use inside the model

            self.ema_model = HARTForT2I(config=config)
            ema_path = os.path.join(model_path, "ema_model.bin")
            self.ema_model.load_state_dict(torch.load(ema_path, map_location=self.device, weights_only=True))
            self.ema_model.to(self.device)

        # Load text encoder and tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path).to(self.device).eval()

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def generate_image(self, prompt):

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
            context_tokens, context_mask, context_position_ids, context_tensor = encode_prompts(
                [prompt],
                self.text_model,
                self.text_tokenizer,
                self.max_token_length,
                llm_system_prompt,
                self.use_llm_system_prompt,
            )

            infer_func = (
                self.ema_model.autoregressive_infer_cfg
                if self.use_ema
                else self.model.autoregressive_infer_cfg
            )

            output_imgs = infer_func(
                B=1,
                label_B=context_tensor,
                cfg=self.cfg,
                g_seed=self.seed,
                more_smooth=self.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )

            img_tensor = output_imgs[0].mul(255).byte().cpu()
            img = img_tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(img)