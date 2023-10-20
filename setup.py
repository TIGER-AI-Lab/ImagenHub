from setuptools import setup, find_packages
import os

__version__ = None

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open(os.path.join("src","imagen_hub","_version.py")) as f:
    exec(f.read(), __version__)

setup(    
    name='imagen_hub',
    version=__version__,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    description='',
    long_description=readme,
    author='Max Ku',
    author_email='m3ku@uwaterloo.ca',
    url='',
    license=license
)