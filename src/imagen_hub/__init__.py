import os
from ._version import __version__

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
if os.environ.get("IMAGE_MODEL_PATH"):
    MODEL_PATH = os.environ.get("IMAGE_MODEL_PATH")

from .infermodels import load, get_model, load_model
