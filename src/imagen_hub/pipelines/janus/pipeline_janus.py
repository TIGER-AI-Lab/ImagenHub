import torch
import numpy as np
import PIL.Image

from .janus_src.models import VLChatProcessor as VLChatProcessor
from .janus_src.models import MultiModalityCausalLM as MultiModalityCausalLM

from .janus_src.janusflow.models import VLChatProcessor as VLChatProcessorFlow
from .janus_src.janusflow.models import MultiModalityCausalLM as MultiModalityCausalLMFlow


from diffusers.models import AutoencoderKL
import torchvision


class JanusPipeline:
    def __init__(self, weight="deepseek-ai/Janus-1.3B", device="cuda"):
        torch.cuda.empty_cache()
        self.device = device
        self.vl_chat_processor = VLChatProcessor.from_pretrained(weight)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.model =  MultiModalityCausalLM.from_pretrained(weight, trust_remote_code=True)
        self.model = self.model.to(torch.bfloat16).to(device).eval()

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        temperature: float = 1.0,
        cfg_weight: float = 5.0,
        img_size: int = 384,
        patch_size: int = 16,
        image_token_num: int = 576
    ):
        # Format the prompt
        conversation = [{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}]
        formatted_prompt = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation, sft_format=self.vl_chat_processor.sft_format, system_prompt=""
        ) + self.vl_chat_processor.image_start_tag

        # Tokenize input
        input_ids = self.tokenizer.encode(formatted_prompt)
        input_ids = torch.LongTensor(input_ids).to(self.device)

        # Prepare tokens
        tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).to(self.device)
        tokens[0, :] = input_ids
        tokens[1, 1:-1] = self.vl_chat_processor.pad_id  # Unconditional branch for CFG

        # Get embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        # Initialize generated tokens
        generated_tokens = torch.zeros((1, image_token_num), dtype=torch.int).to(self.device)

        # Generate image tokens
        for i in range(image_token_num):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            # Classifier-Free Guidance (CFG)
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond, logit_uncond = logits[0::2, :], logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            # Prepare for next step
            next_token = torch.cat([next_token.unsqueeze(dim=1)] * 2, dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # Decode image
        decoded_image = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), shape=[1, 8, img_size // patch_size, img_size // patch_size]
        )
        decoded_image = decoded_image.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        decoded_image = np.clip((decoded_image + 1) / 2 * 255, 0, 255).astype(np.uint8)

        img = PIL.Image.fromarray(decoded_image[0])
        torch.cuda.empty_cache()

        return img
    

class JanusFlowPipeline:
    def __init__(self, weight: str = "deepseek-ai/JanusFlow-1.3B", device: str = "cuda"):
        """
        Initializes the JanusFlow pipeline with specified weights and device.
        """
        torch.cuda.empty_cache()
        self.device = device
        self.vl_chat_processor = VLChatProcessorFlow.from_pretrained(weight)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt = MultiModalityCausalLMFlow.from_pretrained(weight, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).to(device).eval()
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.vae = self.vae.to(torch.bfloat16).to(device).eval()
    
    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        cfg_weight: float = 5.0,
        num_inference_steps: int = 30,
        batchsize: int = 5,
    ):
        """
        Generate images using the JanusFlow model.
        """
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        
        tokens = torch.stack([input_ids] * 2 * batchsize).to(self.device)
        tokens[batchsize:, 1:] = self.vl_chat_processor.pad_id
        inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
        inputs_embeds = inputs_embeds[:, :-1, :]
        
        z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16, device=self.device)
        dt = torch.full_like(z, 1.0 / num_inference_steps, dtype=torch.bfloat16, device=self.device)
        
        attention_mask = torch.ones((2 * batchsize, inputs_embeds.shape[1] + 577), device=self.device, dtype=torch.int)
        attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
        
        for step in range(num_inference_steps):
            z_input = torch.cat([z, z], dim=0)
            t = torch.full((z_input.shape[0],), step / num_inference_steps * 1000., dtype=torch.bfloat16, device=self.device)
            
            z_enc = self.vl_gpt.vision_gen_enc_model(z_input, t)
            z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
            z_emb = self.vl_gpt.vision_gen_enc_aligner(z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1))
            llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

            outputs = self.vl_gpt.language_model.model(inputs_embeds=llm_emb, use_cache=True, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            hidden_states = self.vl_gpt.vision_gen_dec_aligner(self.vl_gpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
            hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
            
            v = self.vl_gpt.vision_gen_dec_model(hidden_states, hs, t_emb)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = cfg_weight * v_cond - (cfg_weight - 1.) * v_uncond
            z = z + dt * v

        decoded_image = self.vae.decode(z / self.vae.config.scaling_factor).sample
        decoded_image = (decoded_image[0].clamp(-1, 1) * 0.5 + 0.5).to(torch.float32)
        img = torchvision.transforms.ToPILImage()(decoded_image)
        torch.cuda.empty_cache()
        return img


