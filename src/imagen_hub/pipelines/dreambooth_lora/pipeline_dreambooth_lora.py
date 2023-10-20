# modified based on https://github.com/Victarry/stable-dreambooth

from imagen_hub.pipelines.dreambooth.utils.dataset import TrainDataset

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.pipelines import StableDiffusionPipeline

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

from diffusers.optimization import get_scheduler
from imagen_hub.pipelines.dreambooth_lora.dreambooth_train import DreamBoothDataset, PromptDataset, \
    import_model_class_from_model_name_or_path, collate_fn, TorchTracemalloc
from PIL import Image
from tqdm import tqdm
import hashlib
import urllib.request
import logging
from pathlib import Path
import os
from transformers import AutoTokenizer, PretrainedConfig
from peft import LoraConfig, get_peft_model, PeftModel
import math
from accelerate.logging import get_logger
import numpy as np


UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]  # , "ff.net.0.proj"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

logger = get_logger(__name__)


class DreamBoothLoraPipeline:
    def __init__(self, device="cuda",
                 subject_name = "dog",
                 identifier = "sks",
                 data_path = None,
                 evaluate_prompt = "photo of a sleeping sks dog",
                 num_epochs: int = 1,
                 train_batch_size: int = 4,
                 learning_rate: float = 1e-5,
                 image_size: int = 512,
                 gradient_accumulation_steps: int = 1,
                 num_train_timesteps: int = 1000,
                 train_guidance_scale: float = 1,
                 sample_guidance_scale: float = 7.5,
                 mixed_precision = 'fp16', save_image_epochs = 1, save_model_epochs = 1,
                 output_dir = None,
                 model_id="CompVis/stable-diffusion-v1-4",
                 num_ref_images=100,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.0,
                 lora_bias='none'
                ):
        self.instance_prompt = "photo of a " + identifier + " " + subject_name
        self.class_prompt = "photo of a " + subject_name
        self.evaluate_prompt = evaluate_prompt
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_timesteps = num_train_timesteps
        self.train_guidance_scale = train_guidance_scale
        self.sample_guidance_scale = sample_guidance_scale
        self.mixed_precision = mixed_precision,
        self.save_image_epochs = save_image_epochs,
        self.save_model_epochs = save_model_epochs,
        self.output_dir = output_dir
        self.model_id = model_id
        self.device = device
        self.data_path = data_path
        self.num_epochs = num_epochs
        self.num_ref_images = num_ref_images
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias

    def train_lora(self):
        logging_dir = Path(self.output_dir, "logs")
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision[0],
            log_with="tensorboard",
            project_dir=logging_dir,
        )
        class_images_dir = Path(self.data_path, "class")
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < self.num_ref_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=None,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = self.num_ref_images - cur_class_images

            sample_dataset = PromptDataset(self.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=4)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        os.makedirs(self.output_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        text_encoder_cls = import_model_class_from_model_name_or_path(self.model_id, revision=None)

        # Load scheduler and models
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )  # DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            self.model_id, subfolder="text_encoder", revision=None
        )
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", revision=None)
        unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", revision=None
        )

        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=UNET_TARGET_MODULES,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
        )

        unet = get_peft_model(unet, config)
        unet.print_trainable_parameters()

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = (unet.parameters())

        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )

        if not os.path.exists(os.path.join(self.data_path, "instance")):
            os.makedirs(os.path.join(self.data_path, "instance"))
        instance_data_dir = os.path.join(self.data_path, "instance")
        class_data_dir = os.path.join(self.data_path, "class")

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=instance_data_dir,
            instance_prompt=self.instance_prompt,
            class_data_root=class_data_dir,
            class_prompt=self.class_prompt,
            tokenizer=tokenizer,
            size=512,
            center_crop=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, True),
            num_workers=1,
        )

        # Scheduler and math around the number of training steps.

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500 * self.gradient_accumulation_steps,
            num_training_steps=1800 * self.gradient_accumulation_steps,
            num_cycles=1,
            power=1.0,
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        num_train_epochs = math.ceil(1800 / num_update_steps_per_epoch)

        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = 1800")
        global_step = 0
        first_epoch = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, 1800), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        print("num_train_epochs: ", num_train_epochs)
        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            with TorchTracemalloc() as tracemalloc:
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(unet):
                        # Convert images to latent space
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                        )
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + prior_loss

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            params_to_clip = (
                                unet.parameters()
                            )
                            accelerator.clip_grad_norm_(params_to_clip, 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    # accelerator.log(logs, step=global_step)

                    if (
                            self.evaluate_prompt is not None
                            and (step + num_update_steps_per_epoch * epoch) % 500 == 0
                    ):
                        logger.info(
                            f"Running validation... \n Generating 2 images with prompt:"
                            f" {self.evaluate_prompt}."
                        )
                        # create pipeline
                        pipeline = DiffusionPipeline.from_pretrained(
                            self.model_id,
                            safety_checker=None,
                            revision=None,
                        )
                        # set `keep_fp32_wrapper` to True because we do not want to remove
                        # mixed precision hooks while we are still training
                        pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
                        pipeline.text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
                        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        generator = torch.Generator(device=accelerator.device).manual_seed(2)
                        images = []
                        for _ in range(2):
                            image = \
                            pipeline(self.evaluate_prompt, num_inference_steps=25, generator=generator).images[0]
                            images.append(image)

                        del pipeline
                        torch.cuda.empty_cache()

                    if global_step >= 1800:
                        break
        unwarpped_unet = accelerator.unwrap_model(unet)
        unwarpped_unet.save_pretrained(
            os.path.join(self.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
        )

        if global_step % 800 == 0 or global_step == 1800:
            if accelerator.is_main_process:
                save_path = os.path.join(self.output_dir, "dreambooth_lora", f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    def get_lora_sd_pipeline(self,
            ckpt_dir, dtype=torch.float16, adapter_name="default"
    ):
        unet_sub_dir = os.path.join(ckpt_dir, "unet")

        pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unet.half()
            pipe.text_encoder.half()

        pipe.to(self.device)
        return pipe
