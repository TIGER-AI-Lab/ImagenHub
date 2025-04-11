class Infinity:
    def __init__(self, model_weight='infinity_2b_reg.pth',vae_weight='infinity_vae_d32_reg.pth',text_encoder_weight='google/flan-t5-xl', device = "cuda"):
    

        from imagen_hub.pipelines.infinity import InfinityPipeline
        self.pipe = InfinityPipeline(model_weight=model_weight,vae_weight=vae_weight,text_encoder_weight=text_encoder_weight, device = device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        self.pipe.set_seed(seed)
        image = self.pipe.generate_image(prompt)
        return image

