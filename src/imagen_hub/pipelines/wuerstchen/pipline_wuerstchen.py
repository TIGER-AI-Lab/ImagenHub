import torch
from .wuerstchen_arc.stages.vqgan import VQModel
from .wuerstchen_arc.models.download import load_checkpoint
from transformers import AutoTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
from .wuerstchen_arc.stages.modules import Paella, EfficientNetEncoder, Prior, DiffNeXt
from .wuerstchen_arc.stages.diffuzz import Diffuzz


class WuerstchenPipeline:

    def __init__(self, device: str = None, type: int = 1):
        """
        Initialize the pipeline.
        :param device: The device we will use.
        :param type: 1 for v1 and 2 for v2
        """
        assert type == 1 or type == 2
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vqmodel = VQModel().to(device)
        self.vqmodel.load_state_dict(load_checkpoint('stage1', self.device))
        self.vqmodel.eval().requires_grad_(False)

        self.clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(
            device).eval().requires_grad_(False)
        self.clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        self.clip_model_b = CLIPTextModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K").eval().requires_grad_(
            False).to(device)
        self.clip_tokenizer_b = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

        self.diffuzz = Diffuzz(device=device)

        if type == 1:
            stageb = load_checkpoint('stage2 v1', self.device)
            self.effnet = EfficientNetEncoder(effnet="efficientnet_v2_l").to(self.device)
            self.effnet.load_state_dict(stageb['effnet_state_dict'])
            self.effnet.eval().requires_grad_(False)
            self.generator = Paella(byt5_embd=1024)
            self.generator.load_state_dict(stageb['state_dict'])
            self.generator.eval().requires_grad_(False).to(self.device)

            stagec = load_checkpoint('stage3 v1', self.device)
            self.model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24).to(self.device)
            self.model.load_state_dict(stagec['ema_state_dict'])
            self.model.eval().requires_grad_(False)
        else:
            stageb = load_checkpoint('stage2 v2', self.device)
            self.effnet = EfficientNetEncoder().to(self.device)
            self.effnet.load_state_dict(stageb['effnet_state_dict'])
            self.effnet.eval().requires_grad_(False)
            self.generator = DiffNeXt()
            self.generator.load_state_dict(stageb['state_dict'])
            self.generator.eval().requires_grad_(False).to(self.device)

            stagec = load_checkpoint('stage3 v2', self.device)
            self.model = Prior(c_in=16, c=1536, c_cond=1280, c_r=64, depth=32, nhead=24).to(self.device)
            self.model.load_state_dict(stagec['ema_state_dict'])
            self.model.eval().requires_grad_(False)

        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        self.generator = torch.compile(self.generator, mode="reduce-overhead", fullgraph=True)

    def decode(self, img_seq):
        return self.vqmodel.decode(img_seq)

    def embed_clip(self, caption, negative_caption="", batch_size=4):
        clip_tokens = self.clip_tokenizer([caption] * batch_size, truncation=True, padding="max_length",
                                          max_length=self.clip_tokenizer.model_max_length, return_tensors="pt").to(
            self.device)
        clip_text_embeddings = self.clip_model(**clip_tokens).last_hidden_state

        clip_tokens_uncond = self.clip_tokenizer([negative_caption] * batch_size, truncation=True, padding="max_length",
                                                 max_length=self.clip_tokenizer.model_max_length,
                                                 return_tensors="pt").to(
            self.device)
        clip_text_embeddings_uncond = self.clip_model(**clip_tokens_uncond).last_hidden_state
        return clip_text_embeddings, clip_text_embeddings_uncond
