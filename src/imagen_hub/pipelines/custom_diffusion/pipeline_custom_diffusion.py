from .custom_diffusion_src.src.retrieve import retrieve
from .custom_diffusion_src.train import train
from .custom_diffusion_src.src.get_deltas import main as get_deltas_main
from .custom_diffusion_src.sample import main as sample_main
import os, PIL
import torch

class CustomDiffusionPipeline():
    def __init__(self, now, concept1, concept2, image_data_concept1, image_data_concept2, config, model_folder, pretrained_ckpt=None,
                 gpus='0', batch_size=1, device="cuda"):
        self.concept1 = concept1
        self.concept2 = concept2
        self.name = f"{self.concept1.replace(' ', '_')}+{self.concept2.replace(' ', '_')}-sdv4"
        self.image_data_concept1 = image_data_concept1
        self.image_data_concept2 = image_data_concept2
        self.pretrained_ckpt = pretrained_ckpt
        self.gpus = gpus
        self.batch_size = batch_size
        self.device = device
        self.real_reg_image_data_concept1 = os.path.join('temp_data', 'real_reg', f'samples_{concept1.replace(" ", "_")}')
        self.real_reg_image_data_concept2 = os.path.join('temp_data', 'real_reg', f'samples_{concept2.replace(" ", "_")}')
        self.config = config
        self.model_folder = model_folder
        self.now = now
        self.output_folder = os.path.join(self.model_folder, f'{self.now}_{self.name}')
        self.delta_ckpt = os.path.join(self.output_folder, 'checkpoints', 'delta_epoch=last.ckpt')
        self.modifier_token = "<new1>+<new2>"

    def train(self):
        print(f"Training custom diffusion and saving to {self.output_folder}")
        retrieve(self.concept1, self.real_reg_image_data_concept1, 200)
        retrieve(self.concept2, self.real_reg_image_data_concept2, 200)
        train(self.now, self.name, self.gpus, self.config, self.model_folder, self.pretrained_ckpt,
              f"<new1> {self.concept1}", self.image_data_concept1,
              os.path.join(self.real_reg_image_data_concept1, 'images.txt'),
              os.path.join(self.real_reg_image_data_concept1, 'caption.txt'),
              f"<new2> {self.concept2}", self.image_data_concept2,
              os.path.join(self.real_reg_image_data_concept2, 'images.txt'),
              os.path.join(self.real_reg_image_data_concept2, 'caption.txt'),
              self.modifier_token, self.batch_size)


    @torch.no_grad()
    def sample(self, prompt, delta_ckpt, pretrained_ckpt) -> PIL.Image.Image:
        if not os.path.exists(delta_ckpt):
            get_deltas_main(self.output_folder, newtoken=2)
        print(f'Generating image for {prompt} from {delta_ckpt} and {pretrained_ckpt}')
        return sample_main(prompt, delta_ckpt, pretrained_ckpt)
