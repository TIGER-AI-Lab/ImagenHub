import os
from typing import Optional

from PIL import Image
from imagen_hub.pipelines.step1xedit import Step1XeditPipeline
from huggingface_hub import snapshot_download

class Step1XEdit:
    """Minimal, user‑friendly wrapper around Step‑1‑X Edit.

    Parameters
    ----------
    model_path : str
        Path to the directory that contains at least
        ``vae.safetensors``, ``step1x-edit-i1258.safetensors`` and the
        Qwen2.5‑VL folder.
    device : str, default "cuda"
        Device used for inference.
    offload : bool, default False
        Enable off‑GPU CPU offloading for very large VRAM‑limited setups.
    quantized : bool, default False
        Load the fp8 quantised weights.
    lora : Optional[str]
        Path to a LoRA file (``*.safetensors``) – if provided the LoRA is
        merged into the model at load time.
    mode : {"flash", "xdit"}
        "flash" for single‑GPU, "xdit" when running with distributed
        USP levels.
    """

    def __init__(
        self,
        weight: str = "stepfun-ai/Step1X-Edit",
        device: str = "cuda",
        offload: bool = False,
        quantized: bool = False,
        lora: Optional[str] = None,
        mode: str = "flash",
    ) -> None:
        
        model_name = weight.split("/")[-1]
        model_path = os.path.join(os.path.dirname(__file__), f"../../../checkpoint/{model_name}")
        model_path = os.path.abspath(model_path)
        qwen_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
        qwen_path = os.path.join(os.path.dirname(__file__), f"../../../checkpoint/Qwen2.5-VL-7B-Instruct")
        qwen_path = os.path.abspath(qwen_path)

        snapshot_download(weight, local_dir=model_path)
        print(f"saved model at {model_path}")

        snapshot_download(qwen_repo, local_dir=qwen_path)
        print(f"saved model at {qwen_path}")

        self.pipeline = Step1XeditPipeline(
            ae_path=os.path.join(model_path, "vae.safetensors"),
            dit_path=os.path.join(model_path, "step1x-edit-i1258.safetensors"),
            qwen2vl_model_path=qwen_path,
            device=device,
            offload=offload,
            quantized=quantized,
            lora=lora,
            mode=mode,
        )

    def infer_one_image(
        self,
        prompt: str,
        src_image: Image.Image = None,
        negative_prompt: str = "",
        num_steps: int = 28,
        cfg_guidance: float = 6.0,
        seed: int = 42,
        strength: float = 0.0,
        size_level: int = 512,
        show_progress: bool = False,
    ) -> Image.Image:
        """Edit or transform *reference* image guided by *prompt*.

        Returns a single ``PIL.Image``.
        """

        images = self.pipeline.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_images=src_image.convert("RGB"),
            num_samples=1,
            num_steps=num_steps,
            cfg_guidance=cfg_guidance,
            seed=seed,
            init_image=None,
            image2image_strength=strength,
            show_progress=show_progress,
            size_level=size_level,
        )
        return images[0]
