import os
import os.path as osp
import random
import argparse
import torch
import numpy as np
import cv2
import urllib.request
from huggingface_hub import snapshot_download
from .infinity_src.tools.run_infinity import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img,
    h_div_w_templates,
    dynamic_resolution_h_w,
)
from PIL import Image

def download_if_not_exists(url: str, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"📥 Downloading from {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Saved to {save_path}")
    else:
        print(f"📁 Found existing checkpoint: {save_path}")

class InfinityPipeline:
    def __init__(self,
                 model_weight='infinity_2b_reg.pth',
                 vae_weight='infinity_vae_d32_reg.pth',
                 text_encoder_weight='google/flan-t5-xl',
                 device="cuda"):
        model_url = 'https://huggingface.co/FoundationVision/infinity/resolve/main/infinity_2b_reg.pth'
        vae_url = 'https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32reg.pth'
        model_path =  os.path.join(os.path.dirname(__file__), f"../../../../checkpoint/infinity/{model_weight}")
        cache_dir = os.path.join(os.path.dirname(__file__), f"../../../../checkpoint/infinity")
        vae_path =  os.path.join(os.path.dirname(__file__), f"../../../../checkpoint/infinity/{vae_weight}")
        download_if_not_exists(model_url, model_path)
        download_if_not_exists(vae_url, vae_path)

        t5_path =  os.path.join(os.path.dirname(__file__), "../../../../checkpoint/")
        snapshot_download(text_encoder_weight, local_dir=f'{t5_path}/flan-t5-xl')
        text_encoder_ckpt = f'{t5_path}/flan-t5-xl'

        self.args = argparse.Namespace(
            pn='1M',
            model_path=model_path,
            cfg_insertion_layer=0,
            vae_type=32,
            vae_path=vae_path,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type='infinity_2b',
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt=text_encoder_ckpt,
            text_channels=2048,
            apply_spatial_patchify=0,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir=cache_dir,
            enable_model_cache=0,
            checkpoint_type='torch',
            seed=0,
            bf16=1
        )
        self.device = device
        self._load_components()
    
    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _load_components(self):
        
        self.text_tokenizer, self.text_encoder = load_tokenizer(self.args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(self.args)
        self.infinity = load_transformer(self.vae, self.args)

    def generate_image(self, prompt, cfg=3, tau=0.5, h_div_w=1.0, enable_positive_prompt=0):

        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        image_tensor = gen_one_img(
            self.infinity,
            self.vae,
            self.text_tokenizer,
            self.text_encoder,
            prompt,
            g_seed=self.seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer],
            vae_type=self.args.vae_type,
            sampling_per_bits=self.args.sampling_per_bits,
            enable_positive_prompt=enable_positive_prompt,
        )

        image_np = image_tensor.cpu().numpy()
        if image_np.shape[0] == 3 and image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        if image_np.dtype == np.float32 or image_np.max() <= 1.0:
            image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_np_rgb)

        return  pil_image

