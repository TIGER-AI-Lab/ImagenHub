import os
import openai
import torch
import PIL
from imagen_hub.utils.image_helper import load_image
from imagen_hub.utils.file_helper import get_file_path, read_key_from_file
import time

class DALLE():
    """
    Class for generating images using the OpenAI DALL-E model.
    Initializes by attempting to get the OpenAI API key from a file or from environment variables.
    """
    def __init__(self):
        try:
            file = get_file_path("openai.env")
            openai.api_key = read_key_from_file(file)
            print(f"Read key from {file}")
        except:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            print("Read key from OPENAI_API_KEY")
 

class DALLE2(DALLE):
    def __init__(self):
        super().__init__()

    def infer_one_image(self, prompt: str, seed: int = 42):
        """
        Infer an image based on the given prompt.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. (Not supported in openai's API)

        Returns:
            PIL.Image.Image: The inferred image.
        """

        try:
            response = openai.Image.create(
                model="dall-e-2",
                prompt=prompt,
                n=1,
                size="512x512",
            )
            image_url = response['data'][0]['url']
            image = load_image(image_url)
        except openai.error.OpenAIError as e:
            print(e.http_status)
            print(e.error)

            # If the image or prompt was rejected, throw a blank (black) image
            if e.error["code"] == "content_policy_violation":
                image = PIL.Image.new(mode="RGB", size=(512,512))

        return image
    
class DALLE3(DALLE):
    def __init__(self):
        super().__init__()

    def infer_one_image(self, prompt: str, seed: int = 42):
        """
        Infer an image based on the given prompt.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. (Not supported in openai's API)

        Returns:
            PIL.Image.Image: The inferred image.
        """
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            image = load_image(image_url)
            time.sleep(2) # The default rate limit for the DALL·E API is 50 images per minute. 
        except openai.error.OpenAIError as e:
            print(e.http_status)
            print(e.error)

            # If the image or prompt was rejected, throw a blank (black) image
            if e.error["code"] == "content_policy_violation":
                image = PIL.Image.new(mode="RGB", size=(1024,1024))

        return image

class StableUnCLIP():
    """
    Class for generating images based on text prompts using the StableUnCLIP pipeline from Huggingface.
    References:
    - https://huggingface.co/docs/diffusers/api/pipelines/unclip
    - https://huggingface.co/docs/diffusers/api/pipelines/stable_unclip
    """
    def __init__(self,
                 device="cuda",
                 prior_model_id="kakaobrain/karlo-v1-alpha",
                 prior_text_model_id = "openai/clip-vit-large-patch14",
                 stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small") -> None:
        """
        Initializes the StableUnCLIP model with specified settings.

        Note:
        For text-to-image, it is recommended to use "stabilityai/stable-diffusion-2-1-unclip-small"
        as it was trained on the same CLIP ViT-L/14 embedding as the Karlo model prior.
        """
        from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
        from diffusers.models import PriorTransformer
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        data_type = torch.float16
        prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)
        prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
        prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
        prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
        prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)
        self.pipe = StableUnCLIPPipeline.from_pretrained(
            stable_unclip_model_id,
            torch_dtype=data_type,
            variant="fp16",
            prior_tokenizer=prior_tokenizer,
            prior_text_encoder=prior_text_model,
            prior=prior,
            prior_scheduler=prior_scheduler,
        ).to(device)

    def infer_one_image(self, prompt: str, seed: int = 42):
        """
        Infer an image based on the given prompt and seed.

        Args:
            prompt (str, optional): The prompt for the image generation. Default is None.
            seed (int, optional): The seed for random generator. Default is 42.

        Returns:
            PIL.Image.Image: The inferred image.
        """

        generator = torch.manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator,
        ).images[0]
        return image
