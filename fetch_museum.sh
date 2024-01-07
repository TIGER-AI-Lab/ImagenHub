#!/bin/bash

# Get the directory of the current script.
SCRIPT_DIR=$(dirname "$0")

# Change to the directory where the script is located.
cd "$SCRIPT_DIR"

git clone https://github.com/ChromAIca/ChromAIca.github.io.git

# Check if 'checkpoints' directory exists in that directory. If not, create it.
if [ ! -d "results" ]; then
    mkdir results
fi

mv ChromAIca.github.io/Museum/* results