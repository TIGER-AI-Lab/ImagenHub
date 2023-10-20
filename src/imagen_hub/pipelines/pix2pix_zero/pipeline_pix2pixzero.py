import torch

from imagen_hub.miscmodels.t5 import T5_Model

class Pix2PixZeroPipeline():
    def __init__(self, device="cuda"):
        self.model = T5_Model(device)
        self.device = device

    def generate_captions(self, input_prompt):
        return self.model.generate_caption_from_text(input_prompt=input_prompt)

    # https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/make_edit_direction.py
    ## convert sentences to sentence embeddings
    def embed_captions(self, l_sentences, tokenizer, text_encoder, device="cuda"):
        """
        embed text captions.
        Params:
            tokenizer: StableDiffusionPipeline tokenizer
            text_encoder: StableDiffusionPipeline text_encoder
        
        Example:
            source_captions = ['a cat eats a worm', 'a stray cat is chased by a cat', 'the cat is so naughty', 'A cat is preparing a meal on the kitchen counter.', 'A black cat is sitting in a chair in a park']
            target_captions = ['dog grooming services in nashville, tn', '- breed standard for dogs', '- wikipedia', 'dog dog dog cat dog', 'dog breeds in california']
            source_embeddings = load_sentence_embeddings(source_captions, tokenizer, text_encoder)
            target_embeddings = load_sentence_embeddings(target_captions, tokenizer, text_encoder)
        """
        with torch.no_grad():
            l_embeddings = []
            for sent in l_sentences:
                text_inputs = tokenizer(
                        sent,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
                l_embeddings.append(prompt_embeds)
        return torch.concatenate(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)