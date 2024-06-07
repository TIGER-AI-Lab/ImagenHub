import torch
import PIL

import os
from pathlib import Path

from diffusers.pipelines import StableDiffusionPipeline


class DreamBooth():
    """
    DreamBooth class for managing and manipulating image diffusion processes.
    """
    def __init__(self,
                 device="cuda",
                 subject_name="dog",
                 identifier="sks",
                 data_path=None,
                 output_dir=None,
                 model_id="CompVis/stable-diffusion-v1-4"
                 ):
        """
        Args:
            device (str, optional): Device for training. Defaults to "cuda".
            subject_name (str, optional): Name of the target subject for the model. Defaults to "dog".
            identifier (str, optional): A unique identifier for the fine-tuning process of DreamBooth. Defaults to "sks".
            data_path (str or None, optional): Path containing data for fine-tuning. Should have a subfolder named "instance"
                with target images or a list of images. Defaults to None.
            output_dir (str or None, optional): Path where model checkpoints will be saved. Defaults to None.
            model_id (str, optional): ID corresponding to the original weights of the diffusion models.
                Defaults to "CompVis/stable-diffusion-v1-4".
        """
        self.device = device
        self.model_path = output_dir
        self.subject_name = subject_name
        self.data_path = data_path
        self.identifier = identifier
        self.model_id = model_id

        self.set_pipe(subject_name=self.subject_name,
                      identifier=self.identifier,
                      data_path=self.data_path,
                      output_dir=self.model_path)

    def set_pipe(self,
                 subject_name=None,
                 identifier=None,
                 data_path=None,
                 output_dir=None):
        """
        Override the pipeline for DreamBooth.

        Args:
            subject_name (str, optional): Name of the target subject for the model. Defaults to current subject name.
            identifier (str, optional): A unique identifier for the fine-tuning process of DreamBooth. Defaults to current identifier.
            data_path (str or None, optional): Path containing data for fine-tuning. Should have a subfolder named "instance"
                with target images or a list of images. Defaults to current data path.
            output_dir (str or None, optional): Path where model checkpoints will be saved. Defaults to current model path.
        """
        from imagen_hub.pipelines.dreambooth.pipeline_dreambooth import DreamBoothPipeline
        #override identifier etc.
        self.model_path = output_dir if output_dir is not None else self.model_path
        self.subject_name = subject_name if subject_name is not None else self.subject_name
        self.data_path = data_path if data_path is not None else self.data_path
        self.identifier = identifier if identifier is not None else self.identifier
        
        self.pipe = DreamBoothPipeline(device=self.device,
                    subject_name=self.subject_name,
                    identifier=self.identifier,
                    data_path=self.data_path,
                    num_epochs= 3,
                    train_batch_size= 8,
                    learning_rate= 1e-5,
                    output_dir=self.model_path,
                    model_id=self.model_id,
                    num_ref_images=100)

    def train(self):
        """Train the DreamBooth model."""
        self.pipe.train()

    def infer_one_image(self, model_path=None, instruct_prompt: str = None, seed: int = 42):
        """
        Inference method for DreamBooth for generating a single image.

        Args:
            model_path (str, optional): Path to the model for inference. Defaults to current model path.
            instruct_prompt (str, optional): Instruction prompt for the inference. Defaults to None.
            seed (int, optional): Seed for randomness. Defaults to 42.

        Returns:
            Image: Generated image.
        """
        model_path = self.model_path if model_path is None else model_path
        model = StableDiffusionPipeline.from_pretrained(
            model_path, use_auth_token=True).to(self.device)
        image = self.pipe.evaluate_one(instruct_prompt, model, seed)

        return image

class DreamBoothLora():

    def __init__(self,
                 device="cuda",
                 subject_name="dog",
                 identifier="sks",
                 data_path=None,
                 output_dir=None,
                 model_id="CompVis/stable-diffusion-v1-4",
                 ):
        self.device = device
        self.model_path = output_dir
        self.subject_name = subject_name
        self.data_path = data_path
        self.identifier = identifier
        self.model_id = model_id

        self.set_pipe(subject_name=self.subject_name,
                      identifier=self.identifier,
                      data_path=self.data_path,
                      output_dir=self.model_path)

    def set_pipe(self,
                 subject_name=None,
                 identifier=None,
                 data_path=None,
                 output_dir=None):
        from imagen_hub.pipelines.dreambooth_lora.pipeline_dreambooth_lora import DreamBoothLoraPipeline
        #override identifier etc.
        self.model_path = output_dir if output_dir is not None else self.model_path
        self.subject_name = subject_name if subject_name is not None else self.subject_name
        self.data_path = data_path if data_path is not None else self.data_path
        self.identifier = identifier if identifier is not None else self.identifier

        self.pipe = DreamBoothLoraPipeline(device=self.device,
                        subject_name=self.subject_name,
                        identifier=self.identifier,
                        data_path=self.data_path,
                        num_epochs=3,
                        train_batch_size=1,
                        learning_rate=1e-5,
                        output_dir=self.model_path,
                        model_id=self.model_id,
                        num_ref_images=100,
                        lora_r=8,
                        lora_alpha=32,
                        lora_dropout=0.0,
                        lora_bias='none'
                    )

    def train(self):
        self.pipe.train_lora()

    def infer_one_image(self, model_path, instruct_prompt: str = None, seed=42):
        pipe = self.pipe.get_lora_sd_pipeline(Path(model_path))
        generator = [torch.Generator(device=self.device).manual_seed(seed)]
        image = pipe(instruct_prompt, generator=generator, num_inference_steps=50, guidance_scale=7.5).images[0]
        return image


class DreamBoothMulti():

    def __init__(self,
                 device="cuda",
                 subject_name=["dog", "barn"],
                 identifier=["<new1>", "<new2>"],
                 data_path=None,
                 output_dir='temp_dreambooth_multi_logs',
                 model_id="CompVis/stable-diffusion-v1-4"):
        """
        param:
            device : device for training
            subject_name :  target subject name
            identifier : the special token for finetuning dreambooth
            data_path : the data path of finetuning, it should contain a subfolder named "instance" with target images
            num_epochs: number of epochs for fine-tuning
            train_batch_size: batch size of fine-tuning
            learning_rate: learning rate of fine-tuning
            image_size: the generated image resolution
            gradient_accumulation_steps: the gradient accumulation steps
            num_train_timesteps : the training steps
            train_guidance_scale: guidance scale at training
            sample_guidance_scale: guidance scale at inference
            mixed_precision: mixed precision mode
            save_model_epochs: save model per epochs
            output_dir: the path to save checkpoints
            model_id: the original weights of diffusion models
            num_ref_images: number of reference images to use in finetuning
        """
        self.device = device
        self.model_path = output_dir
        self.subject_name = subject_name
        self.data_path = data_path
        self.identifier = identifier
        self.model_id = model_id
        self.set_pipe(self.subject_name,
                      self.identifier,
                      self.data_path,
                      self.model_path)

    def set_pipe(self,
                 subject_name=None,
                 identifier=None,
                 data_path=None,
                 output_dir=None):
        """
        Override the pipeline for DreamBooth.

        Args:
            subject_name (str, optional): Name of the target subject for the model. Defaults to current subject name.
            identifier (str, optional): A unique identifier for the fine-tuning process of DreamBooth. Defaults to current identifier.
            data_path (str or None, optional): Path containing data for fine-tuning. Should have a subfolder named "instance"
                with target images or a list of images. Defaults to current data path.
            output_dir (str or None, optional): Path where model checkpoints will be saved. Defaults to current model path.
        """
        from imagen_hub.pipelines.dreambooth.pipeline_dreambooth_multiple_subject import DreamBoothPipelineMulti
        #override identifier etc.
        self.model_path = output_dir if output_dir is not None else self.model_path
        self.subject_name = subject_name if subject_name is not None else self.subject_name
        self.data_path = data_path if data_path is not None else self.data_path
        self.identifier = identifier if identifier is not None else self.identifier

        self.pipe = DreamBoothPipelineMulti(device=self.device,
                                            subject_names=self.subject_name,
                                            identifiers=self.identifier,
                                            data_path=self.data_path,
                                            num_epochs=3,
                                            train_batch_size=2,
                                            learning_rate=1e-5,
                                            output_dir=self.model_path,
                                            model_id=self.model_id,
                                            num_ref_images=100)

    def train(self):
        """
        Not supported currently
        """
        raise NotImplementedError()
        #self.pipe.train()

    def infer_one_image(self, model_path, instruct_prompt: str = None, device="cuda"):
        print(f"Loading model from {model_path}")
        model = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=True).to(device)
        image = self.pipe.evaluate_one(instruct_prompt, model)

        return image
