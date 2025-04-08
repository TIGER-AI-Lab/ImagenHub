# -*- coding: utf-8 -*-
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch
import numpy as np
from .emu3_src.mllm.processing_emu3 import Emu3Processor

class Emu3Pipeline:
    def __init__(self,
                 model_path="BAAI/Emu3-Gen",
                 vq_path="BAAI/Emu3-VisionTokenizer",
                 device="cuda:0",
                 guidance_scale=3.0):

        self.device = device
        self.guidance_scale = guidance_scale

        # Load model and components
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        image_processor = AutoImageProcessor.from_pretrained(vq_path, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(vq_path, device_map=device, trust_remote_code=True).eval()
        self.processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        # Define prompts
        self.positive_suffix = " masterpiece, film grained, best quality."
        self.negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

        # Generation config
        self.gen_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def generate_image(self, prompt, ratio="1:1"):
        full_prompt = prompt + self.positive_suffix

        kwargs = dict(
            mode='G',
            ratio=ratio,
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )

        pos_inputs = self.processor(text=[full_prompt], **kwargs)
        neg_inputs = self.processor(text=[self.negative_prompt], **kwargs)

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)

        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                self.guidance_scale,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to(self.device),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])

        outputs = self.model.generate(
            pos_inputs.input_ids.to(self.device),
            self.gen_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(self.device),
        )

        mm_list = self.processor.decode(outputs[0])
        for im in mm_list:
            if isinstance(im, Image.Image):
                return im

        return None
