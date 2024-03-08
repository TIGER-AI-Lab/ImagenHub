import torch
import PIL
import requests
import os
import datetime

class CustomDiffusion():
    """
    Custom Diffusion pipeline for training and inference.
    """
    def __init__(self,
                 concept1,
                 concept2,
                 image_data_concept1,
                 image_data_concept2,
                 config=None,
                 model_folder='',
                 pretrained_ckpt=None,
                 gpus='0,1,2,3,',
                 batch_size=2,
                 device="cuda"):
        """
        Initializes the CustomDiffusion instance.

        Args:
            concept1 (str): First concept for training.
            concept2 (str): Second concept for training.
            image_data_concept1 (str): Path to image data for the first concept.
            image_data_concept2 (str): Path to image data for the second concept.
            config (str, optional): Configuration file path. Defaults to custom diffusion path.
            model_folder (str, optional): Path to save the model checkpoints. Defaults to 'temp_custom_diffusion_logs'.
            pretrained_ckpt (str, optional): Path to pretrained model checkpoint.
            gpus (str, optional): IDs of GPUs to be used. Defaults to '0,1,2,3,'.
            batch_size (int, optional): Batch size for training. Defaults to 2.
            device (str, optional): Device to use for training. Defaults to "cuda".
        """

        from imagen_hub.pipelines.custom_diffusion import CustomDiffusionPipeline

        if not pretrained_ckpt:
            # TODO rewrite this
            pretrained_ckpt = 'temp/sd-v1-4.ckpt'
            res = requests.get('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt')
            if res.status_code == 200:  # http 200 means success
                with open(pretrained_ckpt, 'wb') as file_handle:
                    file_handle.write(res.content)
        if not config:
            config = os.path.join(os.path.abspath('./'), 'src', 'imagen_hub', 'pipelines', 'custom_diffusion', 'custom_diffusion_src', 'configs', 'custom-diffusion/finetune_joint.yaml')
            config = [config]
        if not model_folder:
            model_folder = 'temp_custom_diffusion_logs'
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.now = now
        self.concept1 = concept1
        self.concept2 = concept2
        self.image_data_concept1 = image_data_concept1
        self.image_data_concept2 = image_data_concept2
        self.config = config
        self.model_folder = model_folder
        self.pretrained_ckpt = pretrained_ckpt
        self.gpus = gpus
        self.batch_size = batch_size
        self.device = device

    def set_pipe(self,
                 concept1,
                 concept2,
                 image_data_concept1,
                 image_data_concept2,
                 model_folder=''):
        """
        Set values to be used in the training pipeline.

        Args:
            concept1 (str): First concept for training.
            concept2 (str): Second concept for training.
            image_data_concept1 (str): Path to image data for the first concept.
            image_data_concept2 (str): Path to image data for the second concept.
            model_folder (str, optional): Path to save the model checkpoints.
        """
        #override values
        self.concept1 = concept1
        self.concept2 = concept2
        self.image_data_concept1 = image_data_concept1
        self.image_data_concept2 = image_data_concept2
        self.model_folder = model_folder

    def train(self):
        """
        Trains the model based on the set pipeline.
        """
        self.pipe = CustomDiffusionPipeline(self.now, self.concept1, self.concept2, self.image_data_concept1, self.image_data_concept2, self.config,
                                            self.model_folder, self.pretrained_ckpt, self.gpus, self.batch_size, self.device)
        self.pipe.train()

    def infer_one_image(self, instruct_prompt, delta_ckpt=None, pretrained_ckpt=None):
        """
        Infer a single image based on a given prompt.

        Args:
            instruct_prompt (str): Instruction prompt for image generation.
            delta_ckpt (str, optional): Checkpoint for the delta model.
            pretrained_ckpt (str, optional): Checkpoint for the pretrained model.

        Returns:
            PIL.Image: Generated image.
        """
        self.pipe = CustomDiffusionPipeline(self.now, self.concept1, self.concept2, self.image_data_concept1, self.image_data_concept2, self.config,
                        self.model_folder, self.pretrained_ckpt, self.gpus, self.batch_size, self.device)
        if not delta_ckpt:
            delta_ckpt = self.pipe.delta_ckpt
        if not pretrained_ckpt and self.pipe:
            pretrained_ckpt = self.pipe.pretrained_ckpt
        images = self.pipe.sample(instruct_prompt, delta_ckpt, pretrained_ckpt)
        return images[0]


if __name__ == "__main__":
    # example
    data_dir = '/shared/xingyu/projects/custom-diffusion/data'
    model_folder = 'temp_logs'
    custom_diffusion = CustomDiffusion('cat', 'car', f'{data_dir}/cat', f'{data_dir}/car',
                                       model_folder=model_folder,
                                       pretrained_ckpt='/shared/xingyu/projects/cross-domain-compositing/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt',
                                    gpus='0,1,2,3,')
    custom_diffusion.train()

    # image = custom_diffusion.infer_one_image("<new1> cat sitting on a <new2> car", '/shared/xingyu/projects/DreamAquarium/temp/custom-diffusion-trained-models/_cat+car-sdv4/checkpoints/delta_epoch=000004.ckpt',
    #                                          '/shared/xingyu/projects/cross-domain-compositing/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt')
    # image.save("<new1> cat sitting on a <new2> car.jpg")
