#!/bin/bash

# Get the directory of the current script.
SCRIPT_DIR=$(dirname "$0")

# Change to the directory where the script is located.
cd "$SCRIPT_DIR"

# Check if 'checkpoints' directory exists in that directory. If not, create it.
if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

# Change directory to 'checkpoints'.
cd checkpoints

# Install git lfs.
git lfs install

# Clone the specified repository.
git clone https://huggingface.co/ImagenHub/ImagenHub_Multi-Concept_IC

echo "Downloaded Multi-Concept_IC to checkpoints."

