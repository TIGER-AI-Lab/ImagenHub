import torch
from torchvision.utils import save_image
from .llamagen_src.tokenizer.tokenizer_image.vq_model import VQ_models
from .llamagen_src.language.t5 import T5Embedder
from .llamagen_src.autoregressive.models.gpt import GPT_models
from .llamagen_src.autoregressive.models.generate import generate
import os
import urllib.request
import numpy as np
from huggingface_hub import snapshot_download
import torchvision.transforms.functional as TF

def download_if_not_exists(url: str, save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"📥 Downloading from {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Saved to {save_path}")
    else:
        print(f"📁 Found existing checkpoint: {save_path}")

class LlamaGenPipeline:
    def __init__(
        self,
        vq_model_name = "VQ-16",
        gpt_model_name = "GPT-XL",
        t5_model_type="flan-t5-xl",
        image_size=256,
        downsample_size=16,
        codebook_size=16384,
        codebook_embed_dim=8,
        gpt_type="t2i",
        cls_token_num=120,
        precision="bf16",
        t5_feature_max_len=120,
        cfg_scale=7.5,
        temperature=1.0,
        top_k=1000,
        top_p=1.0,
        compile_model=False,
        no_left_padding=False,
        device="cuda"
    ):
        self.device = device
        self.no_left_padding = no_left_padding

        self.latent_size = image_size // downsample_size
        self.qzshape = [1, codebook_embed_dim, self.latent_size, self.latent_size]

        vq_ckpt_url = 'https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt'
        gpt_ckpt_url = 'https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage1_256.pt'
        t5_ckpt_url = 'google/flan-t5-xl'
        self.vq_ckpt_path =  os.path.join(os.path.dirname(__file__), "../../../../checkpoint/llamagen/vq_ds16_t2i.pt")
        self.gpt_ckpt_path =  os.path.join(os.path.dirname(__file__), "../../../../checkpoint/llamagen/t2i_XL_stage1_256.pt")
        self.t5_path =  os.path.join(os.path.dirname(__file__), "../../../../checkpoint/llamagen/")
        
        download_if_not_exists(gpt_ckpt_url, self.gpt_ckpt_path)
        download_if_not_exists(vq_ckpt_url, self.vq_ckpt_path)
        snapshot_download(t5_ckpt_url, local_dir=f'{self.t5_path}/{t5_model_type}')

        # Load VQ model
        self.vq_model = VQ_models[vq_model_name](
            codebook_size=codebook_size,
            codebook_embed_dim=codebook_embed_dim
        ).to(self.device)
        self.vq_model.eval()
        vq_ckpt = torch.load(self.vq_ckpt_path, map_location="cpu")
        self.vq_model.load_state_dict(vq_ckpt["model"])
        del vq_ckpt
        print("✅ VQ model loaded.")
        # Load GPT model
        dtype_map = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        self.precision = dtype_map[precision]
        self.gpt_model = GPT_models[gpt_model_name](
            block_size=self.latent_size ** 2,
            cls_token_num=cls_token_num,
            model_type=gpt_type
        ).to(self.device, dtype=self.precision)

        gpt_ckpt = torch.load(self.gpt_ckpt_path, map_location="cpu")
        if "model" in gpt_ckpt:
            state_dict = gpt_ckpt["model"]
        elif "module" in gpt_ckpt:
            state_dict = gpt_ckpt["module"]
        elif "state_dict" in gpt_ckpt:
            state_dict = gpt_ckpt["state_dict"]
        else:
            raise ValueError("Invalid GPT checkpoint format.")
        self.gpt_model.load_state_dict(state_dict, strict=False)
        self.gpt_model.eval()
        del gpt_ckpt

        if compile_model:
            self.gpt_model = torch.compile(self.gpt_model, mode="reduce-overhead", fullgraph=True)
        print("✅ GPT model loaded.")
        # Load T5
        self.t5_model = T5Embedder(
            device=self.device,
            local_cache=True,
            cache_dir=self.t5_path,
            dir_or_name=t5_model_type,
            torch_dtype=self.precision,
            model_max_length=t5_feature_max_len,
        )
        print("✅ T5 Embedder loaded.")
        # Sampling config
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def generate_image(self, prompt: str):
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
            caption_embs, emb_masks = self.t5_model.get_text_embeddings([prompt])

            if not self.no_left_padding:
                emb_masks_flipped = torch.flip(emb_masks, dims=[-1])
                caption_embs_padded = []
                for caption_emb, emb_mask in zip(caption_embs, emb_masks):
                    valid_num = int(emb_mask.sum().item())
                    caption_emb_shifted = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    caption_embs_padded.append(caption_emb_shifted)
                caption_embs = torch.stack(caption_embs_padded)
                emb_masks = emb_masks_flipped

            c_indices = caption_embs * emb_masks[:, :, None]
            c_emb_masks = emb_masks

            # Sampling
            index_sample = generate(
                self.gpt_model, 
                c_indices, 
                self.latent_size ** 2,
                c_emb_masks,
                cfg_scale=self.cfg_scale,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                sample_logits=True
            )

            # Decode
            samples = self.vq_model.decode_code(index_sample, self.qzshape)
            sample = samples[0].cpu().detach()
            sample = (sample + 1) / 2.0
            sample = sample.clamp(0, 1)

            return TF.to_pil_image(sample)
