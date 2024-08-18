from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
                
from imagen_hub.pipelines.textual_inversion.pipeline_textual_inversion import TextualInversionPipeline
from imagen_hub.pipelines.textual_inversion.pipeline_textual_inversion_multiple_subject import TextualInversionPipelineMulti

class TextualInversion():
    """
    A class to handle textual inversion tasks, training, and image generation.
    """
    def __init__(self,
                 device="cuda",
                 what_to_teach='object',
                 placeholder_token='sks',
                 initializer_token='dog',
                 output_dir=None,
                 ):
        self.device = device
        self.what_to_teach = what_to_teach
        self.placeholder_token = placeholder_token
        self.initializer_token = initializer_token
        self.output_dir = output_dir

        self.set_pipe(what_to_teach, placeholder_token, initializer_token, output_dir)

    def set_pipe(self,
                 what_to_teach=None,
                 placeholder_token=None,
                 initializer_token=None,
                 output_dir=None):
        """
        Override the pipeline for TextualInversion.

        Args:
            what_to_teach (str, optional): Information about what the model should be trained on. Defaults to current value.
            placeholder_token (str, optional): Placeholder token for textual inversion. Defaults to current value.
            initializer_token (str, optional): Initial token for setting up textual inversion. Defaults to current value.
            output_dir (str or None, optional): Path where model checkpoints will be saved. Defaults to current output directory.
        """
        # override values if provided
        self.what_to_teach = what_to_teach if what_to_teach is not None else self.what_to_teach
        self.placeholder_token = placeholder_token if placeholder_token is not None else self.placeholder_token
        self.initializer_token = initializer_token if initializer_token is not None else self.initializer_token
        self.output_dir = output_dir if output_dir is not None else self.output_dir

        # Initialize the pipeline
        self.pipe = TextualInversionPipeline(self.what_to_teach,
                                             self.placeholder_token,
                                             self.initializer_token,
                                             self.output_dir)

    def train(self, image_path):
        """Train Textual Inversion."""
        self.pipe.train(image_path)

    def infer_one_image(self, prompt, output_dir, seed=42):
        """
        Inference method for TextualInversion for generating a single image based on a textual prompt.

        Args:
            prompt (str): Textual prompt for the inference.
            output_dir (str): Directory where the pre-trained models and configurations are saved.
            seed (int, optional): Seed for randomness. Defaults to 42.

        Returns:
            Image: Generated image.
        """
        # self.pipe.load_textual_inversion("sd-concepts-library/cat-toy")
        # configs from https://huggingface.co/docs/diffusers/training/text_inversion
        generator = torch.manual_seed(seed)
        pipe = StableDiffusionPipeline.from_pretrained(
            output_dir,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(output_dir, subfolder="scheduler"),
            torch_dtype=torch.float16,
        ).to(self.device)

        num_samples = 1
        image = pipe([prompt] * num_samples,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator).images[0]
        return image

class TextualInversionMulti():
    """
    TODO Refractor this.
    """
    def __init__(self,
                 device="cuda",
                 weight="runwayml/stable-diffusion-v1-5",
                 ):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(weight, torch_dtype=torch.float16, use_safetensors=True).to(self.device)

    def set_pipe(self,
                 paths):
        """
        Load Textual Inversion.
        """
        self.pipe.load_textual_inversion(paths)

    def infer_one_image(self, prompt, seed=42):
        """
        Inference method for TextualInversion for generating a single image based on a textual prompt.

        Args:
            prompt (str): Textual prompt for the inference.
            seed (int, optional): Seed for randomness. Defaults to 42.

        Returns:
            Image: Generated image.
        """
        generator = torch.manual_seed(seed)
        image = self.pipe(prompt,
                        num_inference_steps=50,
                        generator=generator).images[0]
        return image
