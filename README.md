# 🖼️ ImagenHub 
[![arXiv](https://img.shields.io/badge/arXiv-2310.01596-b31b1b.svg)](https://arxiv.org/abs/2310.01596)

[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://imagenhub.readthedocs.io/en/latest/)

[![contributors](https://img.shields.io/github/contributors/TIGER-AI-Lab/ImagenHub)](https://github.com/TIGER-AI-Lab/ImagenHub/graphs/contributors)
[![license](https://img.shields.io/github/license/TIGER-AI-Lab/ImagenHub.svg)](https://github.com/TIGER-AI-Lab/ImagenHub/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/TIGER-AI-Lab/ImagenHub?style=social)](https://github.com/TIGER-AI-Lab/ImagenHub)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTIGER-AI-Lab%2FImagenHub&count_bg=%23C83DB9&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

ImagenHub: Standardizing the evaluation of conditional image generation models 
<br>
ICLR 2024

<div align="center">
<img src="https://github.com/TIGER-AI-Lab/ImagenHub/blob/gh-pages/static/images/banner.png" width="40%">
</div>


ImagenHub is a one-stop library to standardize the inference and evaluation of all the conditional image generation models.
* We define 7 prominent tasks and curate 7 high-quality evaluation datasets for each task. 
* We built a unified inference pipeline to ensure fair comparison. We currently support around 30 models.
* We designed two human evaluation scores, i.e. Semantic Consistency and Perceptual Quality, along with comprehensive guidelines to evaluate generated images. 
* We provide code for visualization, autometrics and Amazon mechanical turk templates.

<div align="center">
 <a href = "https://tiger-ai-lab.github.io/ImagenHub/">[🌐 Project Page]</a> <a href = "https://imagenhub.readthedocs.io/en/latest/index.html">[📘 Documentation]</a> <a href = "https://arxiv.org/abs/2310.01596">[📄 Arxiv]</a> <a href = "https://huggingface.co/ImagenHub">[💾 Datasets]</a> <a href = "https://chromaica.github.io/#imagen-museum">[🏛️ ImagenMuseum]</a> <a href = "https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena">[⚔️ GenAI-Arena]</a>
</div>

## 📰 News
* 2024 Jun 10: [GenAI-Arena](https://arxiv.org/abs/2406.04485v1) Paper is out. It is featured on [Huggingface Daily Papers](https://huggingface.co/papers?date=2024-06-10).
* 2024 Jun 07: ImagenHub is finally on PyPI! Check: [https://pypi.org/project/imagen-hub/](https://pypi.org/project/imagen-hub/)
* 2024 Apr 07: We released [Human evaluation ratings](https://github.com/TIGER-AI-Lab/ImagenHub/tree/main/eval/human_ratings) According to our latest Arxiv paper.
* 2024 Feb 14: Checkout [⚔️ GenAI-Arena ⚔️ : Benchmarking Visual Generative Models in the Wild](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)! 
* 2024 Jan 15: Paper accepted to ICLR 2024! See you in Vienna! 
* 2024 Jan 7: We updated [Human Evaluation Guideline](https://imagenhub.readthedocs.io/en/latest/Guidelines/humaneval.html), [ImagenMuseum Submission](https://imagenhub.readthedocs.io/en/latest/Guidelines/imagenmuseum.html)! Now we welcome researchers to submit their method on ImagenMuseum with minimal effort.
* 2023 Oct 23: Version 0.1.0 released! [ImagenHub’s documentation](https://imagenhub.readthedocs.io/en/latest/index.html) now available!
* 2023 Oct 19: Code Released. Docs under construction.
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

Install from PyPI:
```
pip install imagen-hub
```

Or build from source:
```python
git clone https://github.com/TIGER-AI-Lab/ImagenHub.git
cd ImagenHub
conda env create -f env_cfg/imagen_environment.yml
conda activate imagen
pip install -e .
```

For models like Dall-E, DreamEdit, and BLIPDiffusion, please see [Extra Setup](https://imagenhub.readthedocs.io/en/latest/Guidelines/install.html#environment-management)

For some models (Stable Diffusion, SDXL, CosXL, etc.), you need to login through `huggingface-cli`.
```shell
huggingface-cli login
```

## 👨‍🏫 Get Started [🔝](#-table-of-contents)

### Benchmarking
To reproduce our experiment reported in the paper:

Example for text-guided image generation:
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_t2i.yml
```

Note that the expected output structure would be:
```shell
result_root_folder
└── experiment_basename_folder
    ├── input (If applicable)
    │   └── image_1.jpg ...
    ├── model1
    │   └── image_1.jpg ...
    ├── model2
    │   └── image_1.jpg ...
    ├── ...
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

## 🧠 Philosophy [🔝](#-philosophy-)
By streamlining research and collaboration, ImageHub plays a pivotal role in propelling the field of Image Generation and Editing.

* Purity of Evaluation: We ensure a fair and consistent evaluation for all models, eliminating biases.
* Research Roadmap: By defining tasks and curating datasets, we provide clear direction for researchers. 
* Open Collaboration: Our platform fosters the exchange and cooperation of related technologies, bringing together minds and innovations.

### Implemented Models
We included more than 30 Models in image synthesis. See the full list here:
* Supported Models: https://github.com/TIGER-AI-Lab/ImagenHub/issues/1
* Supported Metrics: https://github.com/TIGER-AI-Lab/ImagenHub/issues/6

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
* [ ] Amazon Mechanical Turk Templates (Coming Soon)

### High quality software engineering standard.

* [X] Documentation
* [X] Type Hints
* [ ] Code Coverage (Coming Soon)

## 🙌 Contributing [🔝](#-table-of-contents)

### For the Community
_**Community contributions are encouraged!**_

ImagenHub is still under development. More models and features are going to be added and we always welcome contributions to help make ImagenHub better. If you would like to contribute, please check out [CONTRIBUTING.md](CONTRIBUTING.md). 

We believe that everyone can contribute and make a difference. Whether it's writing code 💻, fixing bugs 🐛, or simply sharing feedback 💬, your contributions are definitely welcome and appreciated 🙌

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

### For the Researchers:

* Q: How can I use your evaluation method for my method?
* A: Please Refer to [https://imagenhub.readthedocs.io/en/latest/Guidelines/humaneval.html](https://imagenhub.readthedocs.io/en/latest/Guidelines/humaneval.html)

* Q: How can I add my method to ImagenHub codebase?
* A: Please Refer to [https://imagenhub.readthedocs.io/en/latest/Guidelines/custommodel.html](https://imagenhub.readthedocs.io/en/latest/Guidelines/custommodel.html)

* Q: I want to feature my method on [ImagenMuseum](https://chromaica.github.io/#imagen-museum)!
* A: Please Refer to [https://imagenhub.readthedocs.io/en/latest/Guidelines/imagenmuseum.html](https://imagenhub.readthedocs.io/en/latest/Guidelines/imagenmuseum.html)


## 🖊️ Citation [🔝](#-table-of-contents)

Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@inproceedings{
ku2024imagenhub,
title={ImagenHub: Standardizing the evaluation of conditional image generation models},
author={Max Ku and Tianle Li and Kai Zhang and Yujie Lu and Xingyu Fu and Wenwen Zhuang and Wenhu Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=OuV9ZrkQlc}
}
```

```bibtex
@article{ku2023imagenhub,
  title={ImagenHub: Standardizing the evaluation of conditional image generation models},
  author={Max Ku and Tianle Li and Kai Zhang and Yujie Lu and Xingyu Fu and Wenwen Zhuang and Wenhu Chen},
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

