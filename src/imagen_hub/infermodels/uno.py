import os
import random
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

# UNO imports
from ..pipelines.uno.uno_src.uno.flux.pipeline import UNOPipeline, preprocess_ref


class UNO:
    """A lightweight wrapper around **UNOPipeline** that accepts a *single text prompt*
    plus *zero, one, or many* reference images.
    """
    def __init__(
        self,
        weight: str = "flux-dev",
        device: Optional[torch.device] = None,
        offload: bool = False,
        only_lora: bool = True,
        lora_rank: int = 512,
    ) -> None:
        """Load the underlying UNO model.

        Parameters
        ----------
        model_type : str
            One of ``{"flux-dev", "flux-dev-fp8", "flux-schnell"}``.
        device : torch.device | str | None
            Target device. If *None*, chooses "cuda" when available, else "cpu".
        offload : bool
            Whether to off‑load weights to CPU to save GPU memory.
        only_lora : bool
            Load only LoRA adapters instead of full base weights.
        lora_rank : int
            LoRA rank (only if *only_lora* is ``True``).
        """
        # Resolve device
        if device is None:
            from accelerate import Accelerator
            accelerator = Accelerator()
            device = accelerator.device
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Build the underlying UNOPipeline
        self._pipeline = UNOPipeline(
            weight,
            self.device,
            offload,
            only_lora=only_lora,
            lora_rank=lora_rank,
        )
    
    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def infer_one_image(
        self,
        prompt: str,
        input_images: Optional[List[Image.Image]] = None,
        width: int = 512,
        height: int = 512,
        guidance: float = 4.0,
        num_steps: int = 25,
        seed: int = 42,
        ref_size: int = -1,
        pe: str = "d",
    ) -> Image.Image:
        """Generate an image from *prompt* conditioned on *ref_images*.

        Parameters
        ----------
        prompt : str
            The text prompt to guide generation.
        ref_images : list[Image.Image] | None
            Zero, one, or many reference images. They will be resized & encoded.
        width, height : int
            Output resolution (will be padded if not divisible by 8).
        guidance : float
            Text‑to‑image CFG scale.
        num_steps : int
            Diffusion sampling steps.
        seed : int
            Random seed for reproducibility.
        ref_size : int
            Size for reference image preprocessing. ``-1`` → auto (512 if 1 ref else 320).
        pe : str
            Positional encoding type the UNO authors expose ("d", "h", "w", "o").

        Returns
        -------
        PIL.Image.Image
            The generated image.
        """
        self.set_seed(seed)

        # --------------------- reference processing -------------------- #
        input_images = input_images or []  # [] if None
        if ref_size == -1:
            ref_size = 512 if len(input_images) == 1 else 320 if len(input_images) > 1 else 512
        input_imgs_processed = [preprocess_ref(img, ref_size) for img in input_images]

        # --------------------- call underlying UNO --------------------- #
        output_img = self._pipeline(
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            ref_imgs=input_imgs_processed,
            pe=pe,
        )
        return output_img
