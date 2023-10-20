# Dataset Zoo

See image output visualization on [Imagen Museum](https://chromaica.github.io/).

[Dataset page for ImagenHub:](https://huggingface.co/ImagenHub)

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
