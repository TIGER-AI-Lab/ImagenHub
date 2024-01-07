# Submit to ImagenMuseum 

To ensure the transparency of the evaluation experiment, we strongly recommend researchers to submit their methods on [ImagenMuseum](https://chromaica.github.io/#imagen-museum).

ImagenMuseum allows researchers and general public to understand the actual performance of models beyond the metrics.

![](https://i.imgur.com/PSCDTmf.jpg)

You can run your method according to our ImagenHub dataset and save the image as the `sample_{uid}.jpg` (or `sample_{img_id}_{turn_index}` for Mask/Text-guided Editing). 


## Text-guided Image Generation Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Text_to_Image](https://huggingface.co/datasets/ImagenHub/Text_to_Image)
```python
from datasets import load_dataset
infer_dataset = load_dataset("ImagenHub/Text_to_Image")['eval']
```

## Mask-guided Image Editing Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Mask_Guided_Image_Editing](https://huggingface.co/datasets/ImagenHub/Mask_Guided_Image_Editing)
```python
from datasets import load_dataset
infer_dataset = load_dataset("ImagenHub/Mask_Guided_Image_Editing")['filtered']
```

## Text-guided Image Editing Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Text_Guided_Image_Editing](https://huggingface.co/datasets/ImagenHub/Text_Guided_Image_Editing)
```python
from datasets import load_dataset
infer_dataset = load_dataset("ImagenHub/Text_Guided_Image_Editing")['filtered']
```

## Subject-driven Image Generation Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Subject_Driven_Image_Generation](https://huggingface.co/datasets/ImagenHub/Subject_Driven_Image_Generation)
```python
from datasets import load_dataset
train_dataset = load_dataset("ImagenHub/DreamBooth_Concepts")['train']
infer_dataset = load_dataset("ImagenHub/Subject_Driven_Image_Generation")['eval']
```

## Subject-driven Image Editing Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Subject_Driven_Image_Editing](https://huggingface.co/datasets/ImagenHub/Subject_Driven_Image_Editing)
```python
from datasets import load_dataset
train_dataset = load_dataset("ImagenHub/DreamBooth_Concepts")['train']
infer_dataset = load_dataset("ImagenHub/Subject_Driven_Image_Editing")['eval']
```

## Multi-concept Image Composition Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Multi_Subject_Driven_Image_Generation](https://huggingface.co/datasets/ImagenHub/Multi_Subject_Driven_Image_Generation)
```python
from datasets import load_dataset
train_dataset = load_dataset("ImagenHub/Multi_Subject_Concepts")['train']
infer_dataset = load_dataset("ImagenHub/Multi_Subject_Driven_Image_Generation")['train']
```

## Control-guided Image Generation Benchmark Dataset
[https://huggingface.co/datasets/ImagenHub/Control_Guided_Image_Generation](https://huggingface.co/datasets/ImagenHub/Control_Guided_Image_Generation)
```python
from datasets import load_dataset
infer_dataset = load_dataset("ImagenHub/Control_Guided_Image_Generation")['eval']
```

## Submission Guide

You can run your method according to our ImagenHub dataset and save the image as the `sample_{uid}.jpg` (or `sample_{img_id}_{turn_index}` for Mask/Text-guided Editing). 

Your submission format has to be a zip file that contains:

```
YourMethod
--- id.jpg
...
```

Example:
```
DALLE3
--- sample_0.jpg
--- sample_100.jpg
--- sample_101.jpg
--- sample_102.jpg
--- sample_103.jpg
--- sample_104.jpg
...
```

All the experiments has to be done with the requirements:
* Seed fixed to 42
* No hand-crafted prompt engineering
* Person who submit the method has to be one of the authors of corresponding work, unless you integrated the code on ImagenHub codebase.

Feel free to submit your authored method results to m3ku@uwaterloo.ca with Title "ImagenMuseum Submission: {Your Method Name}". We will be updating your result within 1 week.

