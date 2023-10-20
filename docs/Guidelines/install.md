# Installation

## Environment Management
> Since different methods might require a different version of libraries
> We might introduce different conda envs for methods(hopefully there will not be too many)
> * `imagen`: The one env that should be able to run most of the code.
```shell
git clone https://github.com/TIGER-AI-Lab/ImagenHub.git
cd ImagenHub
conda env create -f env_cfg/imagen_environment.yml
conda activate imagen
pip install -e .
```

### Verify the installation
```python
import imagen_hub

print(imagen_hub.__version__) # should print a string
```

## Extra Setup for Text-guided Image Generation

### **Running Dall-E**
* Put your API key in a file `openai.env` in the `keys` folder (or anywhere else, the program will be able to fetch it).
* You can also follow the suggested way from openai (setting up OS environment variable).

## Extra Setup for Subject-driven Image Editing

### **Downloading weights** into `checkpoints` folder
```shell
./download_Subject-Driven_IE_weights.sh
```
### **Running DreamEdit**
DreamEdit depends on SAM / GroundingDINO with CUDA support.
Somehow the installation of SAM/GroundingDINO with GPU support requires some extra work on conda. 
```shell
# make sure you are in a conda environment.
which nvcc # should be something usr/bin/nvcc, otherwise you might skip next line
conda install -c conda-forge cudatoolkit-dev -y 
which nvcc # Now your nvcc path should look something like /home/<name>/anaconda3/envs/<env_name>/bin/nvcc
export BUILD_WITH_CUDA=True
export CUDA_HOME=$CONDA_PREFIX
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### **Running BLIP-Diffusion**
Use the environment `blip_environment.yml`. Current `lavis` is not supported in our default yaml.
```shell
conda env create -f env_cfg/blip_environment.yml
conda activate imagen_blip
```

## Extra Setup for Multi-concept Image Composition

### **Downloading weights** into `checkpoints` folder
```shell
./download_Multi-Concept_IC_weights.sh
```

## Miscellaneous

### Updating the environment
```shell
# To update env (If it doesnt work, delete the environment and create again)
conda env update imagen --file env_cfg/imagen_environment.yml  --prune
```

### Reinstalling the environment
```shell
# If you need to reinstall through deleting the environment and create again:
conda deactivate
conda remove --name imagen --all
conda env create -f env_cfg/imagen_environment.yml
conda activate imagen
```