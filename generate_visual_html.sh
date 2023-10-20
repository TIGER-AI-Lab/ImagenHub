#!/bin/bash

# Get the directory of the current script.
SCRIPT_DIR=$(dirname "$0")

# Change to the directory where the script is located.
cd "$SCRIPT_DIR"

# Define the directory containing the benchmark cfg files
CFG_DIR="benchmark_cfg"

# Iterate over each yml file in the directory and run the command
for cfg in "$CFG_DIR"/*.yml; do
    if [ -f "$cfg" ]; then  # Only process if it's a file
        echo "Running with configuration: $cfg"
        python3 visualize.py --cfg "$cfg"
        echo "-----------------------------------"
    fi
done
