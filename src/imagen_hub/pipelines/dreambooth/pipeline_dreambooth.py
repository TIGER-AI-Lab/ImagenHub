# modified based on https://github.com/Victarry/stable-dreambooth

from imagen_hub.pipelines.dreambooth.utils.dataset import TrainDataset
from imagen_hub.depend.clip_retrieval.clip_client import ClipClient, Modality

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.schedulers import DDPMScheduler
from diffusers.pipelines import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm
import urllib.request
import logging
import os


class DreamBoothPipeline:
    def __init__(self, device="cuda",
                 subject_name = "dog",
                 identifier = "sks",
                 data_path = None,
                 evaluate_prompt = "photo of a sleeping sks dog",
                 num_epochs: int = 5,
                 train_batch_size: int = 1,
                 learning_rate: float = 1e-5,
                 image_size: int = 512,
                 gradient_accumulation_steps: int = 1,
                 num_train_timesteps: int = 1000,
                 train_guidance_scale: float = 1,
                 sample_guidance_scale: float = 7.5,
                 mixed_precision = 'fp16', save_image_epochs = 1, save_model_epochs = 1,
                 output_dir = None,
                 model_id="CompVis/stable-diffusion-v1-4",
                 num_ref_images=100
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

    def pred(self, model, noisy_latent, time_steps, prompt, guidance_scale):
        batch_size = noisy_latent.shape[0]
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = model.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latent_model_input = torch.cat([noisy_latent] * 2) if do_classifier_free_guidance else noisy_latent
        time_steps = torch.cat([time_steps] * 2) if do_classifier_free_guidance else time_steps
        noise_pred = model.unet(latent_model_input, time_steps, encoder_hidden_states=text_embeddings)["sample"]
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def retrieve(self):
        # logging_dir = Path(args.output_dir, "0", args.logging_dir)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger = logging.getLogger('clip_image_retrieval')

        client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name="laion5B-H-14",
            aesthetic_score=9,
            aesthetic_weight=0.5,
            modality=Modality.IMAGE,
            num_images=self.num_ref_images)

        results = client.query(text=self.class_prompt)
        logger.info("{} of {} are retrieved".format(len(results), self.class_prompt))
        for i, result in enumerate(results):
            if not result["url"].endswith(".jpg"):
                continue
            if not os.path.exists(os.path.join(self.data_path, "class")):
                os.makedirs(os.path.join(self.data_path, "class"))
            try:
                urllib.request.urlretrieve(result["url"], os.path.join(self.data_path, "class",  "found" + str(i) + ".jpg"))
            except:
                logger.info("failure")
                continue


    def train_loop(self, model: StableDiffusionPipeline, noise_scheduler, optimizer, train_dataloader):
        # Initialize accelerator and tensorboard logging
        print(self.mixed_precision)
        accelerator = Accelerator(
            mixed_precision=self.mixed_precision[0],
            gradient_accumulation_steps=self.gradient_accumulation_steps,

        )
        print(accelerator.device)
        if accelerator.is_main_process:
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

        global_step = 0

        # Now you train the model
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                instance_imgs, instance_prompt, class_imgs, class_prompt = batch
                imgs = torch.cat((instance_imgs, class_imgs), dim=0).to(self.device)
                prompt = instance_prompt + class_prompt

                # Sample noise to add to the images
                bs = imgs.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=self.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                with torch.no_grad():
                    latents = model.vae.encode(imgs).latent_dist.sample() * 0.18215
                    noise = torch.randn(latents.shape, device=self.device)
                    # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps.cpu().numpy())
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = self.pred(model, noisy_latents, timesteps, prompt, guidance_scale=self.train_guidance_scale)
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.unet.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                # if epoch % self.save_image_epochs[0] == 0 or epoch == self.num_epochs - 1:
                #     self.evaluate(self.evaluate_prompt, epoch, model)

                if (epoch + 1) % self.save_model_epochs[0] == 0 or epoch == self.num_epochs - 1:
                    model.save_pretrained(self.output_dir)


    def make_grid(self, images, rows, cols):
        w, h = images[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid


    def evaluate(self, evaluate_prompt, epoch, pipeline: StableDiffusionPipeline):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        with torch.no_grad():
            # with torch.autocast("cuda"):
            pipeline.to("cuda")
            images = \
            pipeline(evaluate_prompt, num_inference_steps=50, width=self.image_size, height=self.image_size,
                     guidance_scale=self.sample_guidance_scale)
        print(images)
        print(images[0])
        print(len(images))

        # Make a grid out of the images
        image_grid = self.make_grid(images[0], rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.jpg")

    def evaluate_one(self, evaluate_prompt, pipeline: StableDiffusionPipeline, seed: int):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        with torch.no_grad():
            with torch.autocast("cuda"):
                generator = [torch.Generator(device="cuda").manual_seed(seed)]
                images = \
                pipeline([evaluate_prompt], num_inference_steps=50, generator=generator, width=self.image_size, height=self.image_size,
                         guidance_scale=self.sample_guidance_scale)

        return images[0][0]




    def get_dataloader(self,):
        dataset = TrainDataset(self.data_path, self.instance_prompt, self.class_prompt, self.image_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True, drop_last=True,
                                                 pin_memory=True)
        return dataloader

    def train(self,):
        self.retrieve()
        train_dataloader = self.get_dataloader()
        model = StableDiffusionPipeline.from_pretrained(self.model_id, use_auth_token=True,
                                                        cache_dir="./.cache").to(self.device)
        optimizer = torch.optim.AdamW(model.unet.parameters(), lr=self.learning_rate)
        noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps, beta_start=0.00085,
                                        beta_end=0.0120)
        self.train_loop(model, noise_scheduler, optimizer, train_dataloader)
