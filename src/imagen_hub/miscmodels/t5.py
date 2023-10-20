import torch
from PIL import Image

# Flan T5
from transformers import AutoTokenizer, T5ForConditionalGeneration

class T5_Model():

    def __init__(self, device="cuda", weight = "google/flan-t5-xl"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(weight)
        self.model = T5ForConditionalGeneration.from_pretrained(
            weight, device_map="auto", torch_dtype=torch.float16
        ).to(self.device)

    def get_tokenizer(self):
        return self.tokenizer

    def generate_caption_from_text(self, input_prompt):
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids,
            temperature=0.8,
            num_return_sequences=16,
            do_sample=True, max_new_tokens=128, 
            top_k=10
        )
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_text
    