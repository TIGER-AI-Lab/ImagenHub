#!/bin/bash

# Get the directory of the current script.
SCRIPT_DIR=$(dirname "$0")

# Change to the directory where the script is located.
cd "$SCRIPT_DIR"

cd ..
rm -rf docs/source/
pip install -e .
sphinx-apidoc -o docs/source/ src/imagen_hub
cd docs

#pip install sphinx-rtd-theme
#pip install myst_parser
make clean
make html