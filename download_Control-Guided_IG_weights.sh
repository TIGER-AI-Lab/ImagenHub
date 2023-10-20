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

if [ ! -d "ImagenHub_Control-Guided_IG" ]; then
    mkdir ImagenHub_Control-Guided_IG
fi

cd ImagenHub_Control-Guided_IG

if [ ! -d "UniControl" ]; then
    mkdir UniControl
fi

cd UniControl
wget https://storage.googleapis.com/sfr-unicontrol-data-research/unicontrol.ckpt 
#wget https://storage.googleapis.com/sfr-unicontrol-data-research/unicontrol_v1.1.ckpt
#wget https://storage.googleapis.com/sfr-unicontrol-data-research/unicontrol_v1.1.st

echo "Downloaded unicontrol.ckpt to checkpoints/ImagenHub_Control-Guided_IG/UniControl."
