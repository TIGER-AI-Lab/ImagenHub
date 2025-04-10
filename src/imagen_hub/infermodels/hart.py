class Hart:
    def __init__(self, model_weight="mit-han-lab/hart-0.7b-1024px",text_model_path="mit-han-lab/Qwen2-VL-1.5B-Instruct", device = "cuda"):
    

        from imagen_hub.pipelines.hart import HartPipeline
        self.pipe = HartPipeline(weight=model_weight,text_model_path=text_model_path, device = device)

    def infer_one_image(self, prompt: str = None, seed: int = 42):
        self.pipe.set_seed(seed)
        image = self.pipe.generate_image(prompt)
        return image