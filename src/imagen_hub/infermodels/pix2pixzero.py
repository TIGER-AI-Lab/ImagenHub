import torch
import PIL

from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline

class Pix2PixZero():
    def __init__(self, device="cuda", weight="CompVis/stable-diffusion-v1-4"):
        """
        Initialize the Pix2PixZero class.

        Args:
            device (str, optional): The device to run the model on. Defaults to "cuda".
            weight (str, optional): Model weights to load. Defaults to "CompVis/stable-diffusion-v1-4".
        """

        from imagen_hub.pipelines.pix2pix_zero.pipeline_pix2pixzero import Pix2PixZeroPipeline

        captioner_id = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(captioner_id)
        model = BlipForConditionalGeneration.from_pretrained(
            captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )

        self.pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
            weight,
            caption_generator=model,
            caption_processor=processor,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe.inverse_scheduler = DDIMInverseScheduler.from_config(
            self.pipe.scheduler.config)
        # self.pipe.enable_model_cpu_offload()
        self.device = device
        self.pix2pixzeropipe = Pix2PixZeroPipeline()

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed=42):
        """
        Modify the source image based on provided prompts.

        Args:
            src_image (Image): Source image.
            src_prompt (str, optional): Source image caption.
            target_prompt (str, optional): Target image caption.
            instruct_prompt (str, optional): Instruction prompt.
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            PIL.Image: Modified image.
        """
        self.pipe.to(self.device)
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)
        # configs from https://huggingface.co/docs/diffusers/api/pipelines/pix2pix_zero

        caption = self.pipe.generate_caption(src_image)
        inv_latents = self.pipe.invert(
            caption, image=src_image, generator=generator).latents

        source_text = f"Provide a caption for images containing a {src_prompt}. "
        "The captions should be in English and should be no longer than 150 characters."

        target_text = source_text.replace(src_prompt, target_prompt)

        source_captions = self.pix2pixzeropipe.generate_captions(source_text)
        target_captions = self.pix2pixzeropipe.generate_captions(target_text)

        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        src_embeds = self.pix2pixzeropipe.embed_captions(
            source_captions, tokenizer, text_encoder)
        target_embeds = self.pix2pixzeropipe.embed_captions(
            target_captions, tokenizer, text_encoder)

        images = self.pipe(
            caption,
            source_embeds=src_embeds,
            target_embeds=target_embeds,
            num_inference_steps=50,
            cross_attention_guidance_amount=0.15,
            generator=generator,
            latents=inv_latents,
        ).images
        return images[0]
