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

# Check if 'ImagenHub_Subject-Driven_IE' directory exists in 'checkpoints'. If not, create it.
if [ ! -d "ImagenHub_Subject-Driven_IE" ]; then
    mkdir ImagenHub_Subject-Driven_IE
fi

# Change directory to 'ImagenHub_Subject-Driven_IE'.
cd ImagenHub_Subject-Driven_IE

# Install git lfs.
git lfs install

# Clone the specified repository.
git clone https://huggingface.co/ImagenHub/DreamEdit-DreamBooth-Models

echo "Downloaded DreamEdit-DreamBooth-Models to checkpoints/ImagenHub_Subject-Driven_IE."

git clone https://huggingface.co/ImagenHub/DreamBooth-Models

echo "Downloaded DreamBooth-Models to checkpoints/ImagenHub_Subject-Driven_IE."
