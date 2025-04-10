class LlamaGen:
    def __init__(self, vq_model_name="VQ-16",gpt_model_name="GPT-XL", device = "cuda"):
    

        from imagen_hub.pipelines.llamagen import LlamaGenPipeline
        self.pipe = LlamaGenPipeline(vq_model_name=vq_model_name,gpt_model_name=gpt_model_name, device = device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        self.pipe.set_seed(seed)
        image = self.pipe.generate_image(prompt)
        return image

