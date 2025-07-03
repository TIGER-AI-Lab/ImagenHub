import os
import random
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Union

from .bagel_src.data.transforms import ImageTransform
from .bagel_src.data.data_utils import add_special_tokens
from .bagel_src.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from .bagel_src.modeling.qwen2 import Qwen2Tokenizer
from .bagel_src.modeling.autoencoder import load_ae
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from .bagel_src.inferencer import InterleaveInferencer
from huggingface_hub import snapshot_download


class BagelPipeline:
    """
    Supports:
        • generate_image(prompt, images=[...])   # zero, one or many images
        • edit_image(prompt, images=[...])
    """
    def __init__(self, weight: str = "ByteDance-Seed/BAGEL-7B-MoT" , max_mem_per_gpu: str = "40GiB", dtype=torch.bfloat16):
        self.weight = weight
        model_name = self.weight.split("/")[-1]
        self.model_path  = os.path.join(os.path.dirname(__file__), f"../../../../checkpoint/{model_name}")
        self.model_path = os.path.abspath(self.model_path)
        self.dtype = dtype
        self._load_components(max_mem_per_gpu)

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_components(self, max_mem_per_gpu):
        
        save_dir = self.model_path
        repo_id = self.weight
        cache_dir = save_dir + "/cache"

        snapshot_download(cache_dir=cache_dir,
        local_dir=save_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
        )
        print(f"save model at {save_dir}")

        # ----------------------- configs & sub-models --------------------- #
        llm_cfg = Qwen2Config.from_json_file(
            os.path.join(self.model_path, "llm_config.json"))
        llm_cfg.qk_norm = True
        llm_cfg.tie_word_embeddings = False
        llm_cfg.layer_module = "Qwen2MoTDecoderLayer"

        vit_cfg = SiglipVisionConfig.from_json_file(
            os.path.join(self.model_path, "vit_config.json"))
        vit_cfg.rope = False
        vit_cfg.num_hidden_layers -= 1       # MoT trick

        vae_model, vae_cfg = load_ae(
            local_path=os.path.join(self.model_path, "ae.safetensors"))
        

        cfg = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_cfg,
            vit_config=vit_cfg,
            vae_config=vae_cfg,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        # empty-init then sharded load
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_cfg)
            vit_model = SiglipVisionModel(vit_cfg)
            model = Bagel(language_model, vit_model, cfg)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_cfg, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vae_tf = ImageTransform(1024, 512, 16)
        vit_tf = ImageTransform(980, 224, 14)

        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(device_map)

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(self.model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=self.dtype,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )

        model = model.eval()

        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_tf,
            vit_transform=vit_tf,
            new_token_ids=new_token_ids
        )


    def generate_image(
        self,
        prompt: str,
        input_images: Optional[List[Image.Image]] = None,
        think: bool = False,
        inference_hyper: Optional[dict] = None
    ) -> Image.Image:
        """
        Generate an image from a single text prompt, conditioned on zero or more
        images.

        Args:
            prompt: Text prompt.
            input_images: [img1, img2, ...] or None.
            think:  Whether to let the model \"think\" (produce a hidden plan).
            inference_hyper: Dict overriding default generation hyper-params.
        """
        imgs = input_images or []                                # [] if None
        input_seq: List[Union[str, Image.Image]] = imgs + [prompt]

        default_h = dict(
            max_think_token_n=1000,
            do_sample=False,
            text_temperature=0.3,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global"
        )
        if inference_hyper:
            default_h.update(inference_hyper)

        out = self.inferencer.interleave_inference(
            input_seq,
            think=think,
            understanding_output=False,
            **default_h
        )

        # return the last PIL image in output list
        for o in reversed(out):
            if isinstance(o, Image.Image):
                return o
        raise RuntimeError("No image generated!")

    def edit_image(
        self,
        prompt: str,
        input_images: List[Image.Image],
        think: bool = False,
        inference_hyper: Optional[dict] = None
    ) -> Image.Image:
        """
        Edit (or continue) generation conditioned on one or more images.

        The **first** image in `images` is treated as the base to edit; any
        additional images act as extra conditioning context.

        Args:
            prompt: Text instruction for the edit.
            input_images: List of PIL images (at least one).
            think:  Enable internal \"planning\".
            inference_hyper: Dict overriding default editing hyper-params.
        """
        if not input_images:
            raise ValueError("edit_image requires at least one image.")

        input_seq: List[Union[str, Image.Image]] = input_images + [prompt]

        default_h = dict(
            max_think_token_n=1000,
            do_sample=False,
            text_temperature=0.3,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel"
        )
        if inference_hyper:
            default_h.update(inference_hyper)

        out = self.inferencer.interleave_inference(
            input_seq,
            think=think,
            understanding_output=False,
            **default_h
        )

        for o in reversed(out):
            if isinstance(o, Image.Image):
                return o
        raise RuntimeError("No image generated!")