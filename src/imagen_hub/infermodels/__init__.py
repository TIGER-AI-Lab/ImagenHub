
# ==========================================================
# Text-to-Image Generation
from .sd import SD, OpenJourney
from .sdxl import SDXL
from .deepfloydif import DeepFloydIF
from .dalle import DALLE, StableUnCLIP
from .unidiffuser import UniDiffuser
from .kandinsky import Kandinsky

# ==========================================================
# Text-guided Image Editing
from .diffedit import DiffEdit
from .imagic import Imagic
from .instructpix2pix import InstructPix2Pix, MagicBrush
from .prompt2prompt import Prompt2prompt
from .text2live import Text2Live
from .sdedit import SDEdit
from .cyclediffusion import CycleDiffusion
from .pix2pixzero import Pix2PixZero

# ==========================================================
# Mask-guided Image Editing
from .blended_diffusion import BlendedDiffusion
from .glide import Glide
from .sd import SDInpaint
from .sdxl import SDXLInpaint

# ==========================================================
# Subject-Driven Image Editing
from .photoswap import PhotoSwap
from .blip_diffusion import BLIPDiffusion_Edit
from .dreamedit import DreamEdit

# ==========================================================
# Subject-Driven Image Generation
from .dreambooth import DreamBooth, DreamBoothLora
from .textual_inversion import TextualInversion
from .blip_diffusion import BLIPDiffusion_Gen

# ==========================================================
# Multi-Subject-Driven Image Generation
from .custom_diffusion import CustomDiffusion
from .dreambooth import DreamBoothMulti
from .textual_inversion import TextualInversionMulti

# ==========================================================
# Control-Guided Image Generation / Custom Condition Guided Image Generation
from .control_net import ControlNet
from .unicontrol import UniControl

# ==========================================================
# Misc
# ==========================================================
#from .crossdomain_composition import CrossDomainComposition
#from .styledrop import StyleDrop

import sys
from functools import partial

def get_model(model_name: str = None, init_with_default_params: bool = True):
    """
    Retrieves a model class or instance by its name.

    Args:
        model_name (str): Name of the model class. Triggers an error if the module name does not exist.
        init_with_default_params (bool, optional): If True, returns an initialized model instance; otherwise, returns
            the model class. Default is True. If set to True, be cautious of potential ``OutOfMemoryError`` with insufficient CUDA memory.

    Returns:
        model_class or model_instance: Depending on ``init_with_default_params``, either the model class or an instance of the model.

    Examples::
        initialized_model = infermodels.get_model(model_name='DiffEdit', init_with_default_params=True)

        uninitialized_model = infermodels.get_model(model_name='DiffEdit', init_with_default_params=False)
        initialized_model = uninitialized_model(device="cuda", weight="stabilityai/stable-diffusion-2-1")
    """

    if not hasattr(sys.modules[__name__], model_name):
        raise ValueError(f"No model named {model_name} found in infermodels.")

    model_class = getattr(sys.modules[__name__], model_name)
    if init_with_default_params:
        model_instance = model_class()
        return model_instance
    return model_class

load_model = partial(get_model, init_with_default_params=True)
load = partial(get_model)
