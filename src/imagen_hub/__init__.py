import os
from ._version import __version__
import sys

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
if os.environ.get("IMAGE_MODEL_PATH"):
    MODEL_PATH = os.environ.get("IMAGE_MODEL_PATH")


# Get the absolute path to SoM
som_path = os.path.join(os.path.dirname(__file__), "SoM")

# Add SoM to sys.path if not already there
if som_path not in sys.path:
    sys.path.insert(0, som_path)
from .infermodels import load, get_model, load_model
