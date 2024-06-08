from setuptools import setup, find_packages
import os

# Initialize an empty dictionary for version information
version_info = {}
with open(os.path.join("src", "imagen_hub", "_version.py")) as f:
    exec(f.read(), version_info)

# Read the content of README.md and LICENSE files
with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(    
    name='imagen_hub',
    version=version_info['__version__'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description="ImagenHub is a one-stop library to standardize the inference and evaluation of all the conditional image generation models.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Max Ku',
    author_email='m3ku@uwaterloo.ca',
    url='https://github.com/TIGER-AI-Lab/ImagenHub',
    license=license
)
