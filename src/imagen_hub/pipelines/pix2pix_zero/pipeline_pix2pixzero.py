import torch

from imagen_hub.miscmodels.t5 import T5_Model

class Pix2PixZeroPipeline():
    """
    Provides a pipeline for caption generation and embedding using the T5 Model.

    Attributes:
        model (T5_Model): Instance of the T5_Model for caption generation.
        device (str): Device the pipeline is running on (e.g., "cuda" or "cpu").

    Args:
        device (str, optional): Specifies the device to run on. Defaults to "cuda".
    """
    def __init__(self, device="cuda"):
        self.model = T5_Model(device)
        self.device = device

    def generate_captions(self, input_prompt):
        """
        Generates a caption based on the provided input prompt.

        Args:
            input_prompt (str): The input text prompt for caption generation.

        Returns:
            str: Generated caption.
        """
        return self.model.generate_caption_from_text(input_prompt=input_prompt)

    # https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/make_edit_direction.py
    ## convert sentences to sentence embeddings
    def embed_captions(self, l_sentences, tokenizer, text_encoder, device="cuda"):
        """
        Converts a list of sentences into sentence embeddings.

        Args:
            l_sentences (List[str]): List of sentences to be embedded.
            tokenizer: StableDiffusionPipeline tokenizer.
            text_encoder: StableDiffusionPipeline text_encoder.
            device (str, optional): Device to perform computations on. Defaults to "cuda".

        Returns:
            torch.Tensor: Sentence embeddings.

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
