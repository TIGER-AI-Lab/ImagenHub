import base64
import os
import PIL.Image
try:
    from google import genai
    from google.genai import types
    import tempfile
except ImportError:
    print("Warning: Failed to import google.genai. Please install it with 'pip install google-genai'")
    print("The Gemini2NativeEdit class will not be available.")

from imagen_hub.utils.image_helper import decode_base64_to_image

class Gemini2NativeEdit:
    """
    A wrapper around the Gemini API for guided image transformation.

    This class uses the Gemini API to transform an image based on an instruction prompt.
    """
    def __init__(self, device="cuda", model="gemini-2.0-flash-exp", api_key=None):
        """
        You need to set up the Gemini API key in the environment variable. <GEMINI_API_KEY>
        Attributes:
            client (genai.Client): The Gemini API client for image transformation.

        Args:
            device (str, optional): Device on which the pipeline runs. Defaults to "cuda".
            api_key (str, optional): API key for Gemini. Defaults to environment variable.
        """
        self.device = device
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        )
        self.model_name = model

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42):
        """
        Modifies the source image based on the provided instruction prompt.

        Args:
            src_image (PIL.Image.Image): Source image in RGB format.
            src_prompt (str): Original image prompt (not used).
            target_prompt (str): Target image prompt (not used).
            instruct_prompt (str): Caption for editing the image.
            seed (int, optional): Seed for random generator (not used). Defaults to 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        # Create temporary file with .png extension
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_input:
            input_path = temp_input.name

        # Save source image
        src_image.save(input_path)

        files = [
            self.client.files.upload(file=input_path),
        ]

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=files[0].uri,
                        mime_type=files[0].mime_type,
                    ),
                    types.Part.from_text(text=instruct_prompt),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_modalities=[
                "image",
                "text",
            ],
            response_mime_type="text/plain",
        )

        try:
            tries = 5
            while tries > 0:
                # Get response using streaming to handle large outputs
                for chunk in self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                        continue
                    
                    if chunk.candidates[0].content.parts[0].inline_data:
                        result_image = decode_base64_to_image(chunk.candidates[0].content.parts[0].inline_data.data)
                        return result_image
                    else:
                        print(chunk.text)

                tries -= 1
                if tries > 0:
                    print(f"No image generated, retrying... ({tries} attempts remaining)")
                    continue
                
            error_msg = "No image was generated after 5 attempts, returning black image"
            print(f"Error: {error_msg}")
            # Return black image after all retries failed
            black_image = PIL.Image.new('RGB', (512, 512), color='black')
            return black_image

        finally:
            # Clean up temp file
            os.unlink(input_path)
