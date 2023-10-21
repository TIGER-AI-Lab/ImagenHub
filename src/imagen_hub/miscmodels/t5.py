import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

class T5_Model():
    """
    A class to represent the T5 (Text-to-Text Transfer Transformer) model.
    """
    def __init__(self, device="cuda", weight = "google/flan-t5-xl"):
        """
        Initialize a T5_Model object with the specified device and weight.
        
        Args:
            device (str, optional): The device on which the model will run. Defaults to "cuda".
            weight (str, optional): Pretrained model weights to load. Defaults to "google/flan-t5-xl".
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(weight)
        self.model = T5ForConditionalGeneration.from_pretrained(
            weight, device_map="auto", torch_dtype=torch.float16
        ).to(self.device)

    def get_tokenizer(self):
        """
        Returns the tokenizer associated with the T5 model.
        
        Returns:
            transformers.AutoTokenizer: Tokenizer associated with the T5 model.
        """
        return self.tokenizer

    def generate_caption_from_text(self, input_prompt):
        """
        Generate captions for the provided input text.
        
        Args:
            input_prompt (str): The text input for which captions are to be generated.
            
        Returns:
            List[str]: List of generated captions.
        """
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
