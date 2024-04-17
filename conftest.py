import os
import sys

# Calculate the path to the root of the project
root_path = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the sys.path
sys.path.append(root_path)

sys.path.append(os.path.join(root_path, "src"))
