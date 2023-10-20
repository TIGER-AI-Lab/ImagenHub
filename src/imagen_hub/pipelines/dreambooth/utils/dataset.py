# credit to https://github.com/Victarry/stable-dreambooth

from typing import List
from PIL import Image
from torch.utils import data
from pathlib import Path
from torchvision import transforms
import os

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
    '.tif',
    '.TIF',
    '.tiff',
    '.TIFF',
]


def is_image_file(file: Path):
    return file.suffix in IMG_EXTENSIONS


def make_dataset(dir, max_dataset_size=float("inf")) -> List[Path]:
    images = []
    root = Path(dir)
    root.mkdir(parents=True, exist_ok=True)
    assert root.is_dir(), '%s is not a valid directory' % dir

    for file in root.rglob('*'):
        if is_image_file(file):
            try:
                Image.open(file).convert('RGB')
                images.append(file)
            except:
                continue
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 return_paths=False,
                 return_dict=False,
                 sort=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if sort:
            imgs = sorted(imgs)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.return_dict = return_dict
        self.loader = loader

    def __getitem__(self, index):
        index = index % len(self)
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, str(path)
        else:
            if self.return_dict:
                return {'images': img}
            else:
                return img

    def __len__(self):
        return len(self.imgs)


class MergeDataset(data.Dataset):
    def __init__(self, *datasets):
        """Merge multiple datasets to one dataset, and each time retrives a combinations of items in all sub datasets.
        """
        self.datasets = datasets
        self.sizes = [len(dataset) for dataset in datasets]
        print('dataset size', self.sizes)

    def __getitem__(self, indexs: List[int]):
        return tuple(dataset[idx] for idx, dataset in zip(indexs, self.datasets))

    def __len__(self):
        return max(self.sizes)


class TrainDataset(data.Dataset):
    def __init__(self, data_path, instance_prompt, class_prompt, image_size):
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        path_instance = os.path.join(data_path, 'instance')
        path_class = os.path.join(data_path, 'class')
        self.data1 = ImageFolder(path_instance, self.transform)  # instance dataset
        self.data2 = ImageFolder(path_class, self.transform)  # class dataset

        self.sizes = [len(self.data1), len(self.data2)]

    def __getitem__(self, index):
        img1 = self.data1[index]
        img2 = self.data2[index]
        return img1, self.instance_prompt, img2, self.class_prompt

    def __len__(self):
        return max(self.sizes)


class MultiTrainDataset(data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, data_path, subject_names, instance_prompts, class_prompts, image_size):
        self.instance_prompts = instance_prompts
        self.class_prompts = class_prompts
        self.subject_names = subject_names
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.data1 = []
        self.data2 = []
        for i, subject_name in enumerate(subject_names):
            path_instance = os.path.join(data_path, 'instance', 'data', subject_name)
            path_class = os.path.join(data_path, 'class', subject_name)
            self.data1.append(ImageFolder(path_instance, self.transform))  # instance dataset
            self.data2.append(ImageFolder(path_class, self.transform))  # class dataset

        self.sizes = [len(self.data1[0]) + len(self.data1[1]), len(self.data2[0])+ len(self.data2[1])]


    def __len__(self):
        return max(self.sizes)

    def __getitem__(self, index):
        if index < len(self.data1[0]):
            img1 = self.data1[0][index]
            instance_prompt = self.instance_prompts[0]
            img2 = self.data2[0][index]
            class_prompt = self.class_prompts[0]
        else:
            index = index - len(self.data1[0])
            img1 = self.data1[1][index]
            instance_prompt = self.instance_prompts[1]
            img2 = self.data2[1][index]
            class_prompt = self.class_prompts[1]
        return img1, instance_prompt, img2, class_prompt
