# 🖼️ ImagenHub 
[![arXiv](https://img.shields.io/badge/arXiv-2310.01596-b31b1b.svg)](https://arxiv.org/abs/2310.01596)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://imagenhub.readthedocs.io/en/latest/)
[![contributors](https://img.shields.io/github/contributors/TIGER-AI-Lab/ImagenHub)](https://github.com/TIGER-AI-Lab/ImagenHub/graphs/contributors)
[![open issues](https://isitmaintained.com/badge/open/TIGER-AI-Lab/ImagenHub.svg)](https://github.com/TIGER-AI-Lab/ImagenHub/issues)
[![pull requests](https://img.shields.io/github/issues-pr/TIGER-AI-Lab/ImagenHub?color=0088ff)](https://github.com/TIGER-AI-Lab/ImagenHub/pulls)
[![license](https://img.shields.io/github/license/TIGER-AI-Lab/ImagenHub.svg)](https://github.com/TIGER-AI-Lab/ImagenHub/blob/main/LICENSE)

<div align="center">
<img src="https://github.com/TIGER-AI-Lab/ImagenHub/blob/gh-pages/static/images/banner.png" width="40%">
</div>


ImagenHub is a one-stop library to standardize the inference and evaluation of all the conditional image generation models.
* We define 7 prominent tasks and curate 7 high-quality evaluation datasets for each task. 
* We built a unified inference pipeline to ensure fair comparison. We currently support around 30 models.
* We designed two human evaluation scores, i.e. Semantic Consistency and Perceptual Quality, along with comprehensive guidelines to evaluate generated images. 
* We provide code for visualization, autometrics and Amazon mechanical turk templates.

<div align="center">
 🔥 🔥 🔥 Check out our <a href = "https://tiger-ai-lab.github.io/ImagenHub/">[Project Page]</a> and <a href = "https://imagenhub.readthedocs.io/en/latest/index.html">[Documentation]</a> for more results and analysis!
</div>

## 📰 News
* 2023 Oct 19: Code Released. Docs (under construction) available on [ImagenHub’s documentation](https://imagenhub.readthedocs.io/en/latest/index.html) !
* 2023 Oct 13: We released [Imagen Museum](https://chromaica.github.io/#imagen-museum), a visualization page for all models from ImagenHub!
* 2023 Oct 4: Our paper is featured on [Huggingface Daily Papers](https://huggingface.co/papers?date=2023-10-04)!
* 2023 Oct 2: Paper available on [Arxiv](https://arxiv.org/abs/2310.01596). Code coming Soon!

## 📄 Table of Contents

- [🛠️ Installation](#%EF%B8%8F-installation-)
- [👨‍🏫 Get Started](#-get-started-)
- [📘 Documentation](#-documentation-)
- [🧠 Philosophy](#-philosophy-)
- [🙌 Contributing](#-contributing-)
- [🖊️ Citation](#%EF%B8%8F-citation-)
- [🤝 Acknowledgement](#-acknowledgement-)
- [🎫 License](#-license-)

## 🛠️ Installation [🔝](#-table-of-contents)
```python
git clone https://github.com/TIGER-AI-Lab/ImagenHub.git
cd ImagenHub
conda env create -f env_cfg/imagen_environment.yml
conda activate imagen
pip install -e .
```

For models like Dall-E, DreamEdit, and BLIPDiffusion, please see [Extra Setup](https://imagenhub.readthedocs.io/en/latest/Guidelines/install.html#environment-management)

## 👨‍🏫 Get Started [🔝](#-table-of-contents)

### Benchmarking
To reproduce our experiment reported in the paper:

Example for text-guided image generation:
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_t2i.yml
```
Then after running the experiment, you can run
```shell
python3 visualize.py --cfg benchmark_cfg/ih_t2i.yml
```
to produce a `index.html` file for visualization.

The file would look like something like this. We hosted our experiment results on [Imagen Museum](https://chromaica.github.io/#imagen-museum).
<img src="https://i.imgur.com/0uOMhtT.png" width="512" />


### Infering one model
```python
import imagen_hub

model = imagen_hub.load("SDXL")
image = model.infer_one_image(prompt="people reading pictures in a museum, watercolor", seed=1)
image
```
<img src="https://i.imgur.com/ruU0BJ0.jpg" width="256" />

### Running Metrics
```python
from imagen_hub.metrics import MetricLPIPS
from imagen_hub.utils import load_image, save_pil_image, get_concat_pil_images

def evaluate_one(model, real_image, generated_image):
  score = model.evaluate(real_image, generated_image)
  print("====> Score : ", score)

image_I = load_image("https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_102724_1.jpg")
image_O = load_image("https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_102724_1.jpg")
show_image = get_concat_pil_images([image_I, image_O], 'h')

model = MetricLPIPS()
evaluate_one(model, image_I, image_O) # ====> Score :  0.11225218325853348

show_image
```
<img src="https://i.imgur.com/af8CB4c.jpg" width="512" />




## 📘 Documentation [🔝](#-table-of-contents)
The tutorials and API documentation are hosted on [imagenhub.readthedocs.io](https://imagenhub.readthedocs.io/en/latest/index.html).

* [Write your own benchmark](https://imagenhub.readthedocs.io/en/latest/Guidelines/custombenchmark.html)

## 🧠 Philosophy [🔝](#-philosophy-)
By streamlining research and collaboration, ImageHub plays a pivotal role in propelling the field of Image Generation and Editing.

* Purity of Evaluation: We ensure a fair and consistent evaluation for all models, eliminating biases.
* Research Roadmap: By defining tasks and curating datasets, we provide clear direction for researchers. 
* Open Collaboration: Our platform fosters the exchange and cooperation of related technologies, bringing together minds and innovations.

### Implemented Models

|        Method     	         |   Venue  	    |            Type           	|
|:---------------------------:|:-------------:|:-------------------------:	|
|       Stable Diffusion   	        |  - 	   | Text-To-Image Generation 	|
|       Stable Diffusion XL   	        |  arXiv'23 	   | Text-To-Image Generation 	|
|       DeepFloyd-IF   	        |  - 	   | Text-To-Image Generation 	|
|       OpenJourney   	        |  - 	   | Text-To-Image Generation 	|
|       Dall-E   	        |  - 	   | Text-To-Image Generation 	|
|       Kandinsky  	        |  - 	   | Text-To-Image Generation 	|
|       MagicBrush   	        |  arXiv'23 	   | Text-guided Image Editing 	|
|      InstructPix2Pix 	      |   CVPR'23 	   | Text-guided Image Editing 	|
|        DiffEdit    	        |  ICLR'23 	   | Text-guided Image Editing 	|
|         Imagic    	         |   CVPR'23	   | Text-guided Image Editing 	|
|     CycleDiffusion    	     |  ICCV'23 	   | Text-guided Image Editing 	|
|         SDEdit    	         |   ICLR'22 	   | Text-guided Image Editing 	|
|    Prompt-to-Prompt    	    |   ICLR'23 	   | Text-guided Image Editing 	|
|          Text2Live          |   ECCV'22 	   | Text-guided Image Editing 	|
|        Pix2PixZero 	        | SIGGRAPH'23 	 | Text-guided Image Editing 	|
|         GLIDE    	          |   ICML'22 	   | Mask-guided Image Editing 	|
|      Blended Diffusion      |   CVPR'22 	   | Mask-guided Image Editing 	|
| Stable Diffusion Inpainting |      - 	      | Mask-guided Image Editing 	|
| Stable Diffusion XL Inpainting |      - 	      | Mask-guided Image Editing 	|
|     TextualInversion        | ICLR'23  | Subject-driven Image Generation|
|       BLIP-Diffusion     |   arXiv'23    | Subject-Driven Image Generation|
|         DreamBooth(+ LoRA)          |    CVPR'23    | Subject-Driven Image Generation|
|       Photoswap    	        |  arXiv'23 	   | Subject-Driven Image Editing 	|
|       DreamEdit    	        |  arXiv'23 	   | Subject-Driven Image Editing 	|
|      Custom Diffusion       |    CVPR'23    | Multi-Subject-Driven Generation|
|         ControlNet          |   arXiv'23    | Control-guided Image Generation|
|         UniControl          |   arXiv'23    | Control-guided Image Generation|

### Comprehensive Functionality
* [X] Common Metrics for GenAI
* [X] Visualization tool
* [ ] Amazon Mechanical Turk Templates (TBA)

### High quality software engineering standard.

* [X] Documentation
* [X] Type Hints
* [ ] Code Coverage (TBA)

## 🙌 Contributing [🔝](#-table-of-contents)

_**Community contributions are encouraged!**_

ImagenHub is still under development. More models and features are going to be added and we always welcome contributions to help make ImagenHub better. If you would like to contribute, please check out [CONTRIBUTING.md](CONTRIBUTING.md). 

Please also feel free to open a PR if you think something is missing here. Always welcome feedback and suggestions. Just open an issue!

## 🖊️ Citation [🔝](#-table-of-contents)

Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@article{ku2023imagenhub,
  title={ImagenHub: Standardizing the evaluation of conditional image generation models},
  author={Max Ku, Tianle Li, Kai Zhang, Yujie Lu, Xingyu Fu, Wenwen Zhuang, Wenhu Chen},
  journal={arXiv preprint arXiv:2310.01596},
  year={2023}
}
```

## 🤝 Acknowledgement [🔝](#-table-of-contents)

Please refer to [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)

## 🎫 License [🔝](#-table-of-contents)

This project is released under the [License](LICENSE).


## ⭐ Star History [🔝](#-table-of-contents)

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/ImagenHub&type=Date)](https://star-history.com/#TIGER-AI-Lab/ImagenHub&Date)

