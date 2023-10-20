import os, json
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, Dataset, IterableDataset
import datasets

images = {}
img_dir = '/shared/xingyu/projects/custom-diffusion/data'
for concept in os.listdir(img_dir):
    if '.txt' in concept: continue
    images[concept] = []
    for image in os.listdir(os.path.join(img_dir, concept)):
        image_path = os.path.join(img_dir, concept, image)
        image = Image.open(image_path)
        im2 = image.copy()
        images[concept].append(im2)
        image.close()
to_save_images = []
for key, values in images.items():
    for value in values:
        to_save_images.append({'image': value, 'concept': key.replace('_', ' ')})
data = []
test_prompts = json.load(open('/shared/xingyu/projects/custom-diffusion/test_prompts.json', 'r'))
for key in tqdm(test_prompts):
    prompts = test_prompts[key]
    concepts = key.split('_')
    if len(concepts) == 3:
        concept1 = concepts[0] + ' ' + concepts[1]
        concept2 = concepts[2]
    else:
        concept1 = concepts[0]
        concept2 = concepts[1]
    for concept in [concept1, concept2]:
        img_dir = '/shared/xingyu/projects/custom-diffusion/data'
        if concept.replace(' ', '_') not in os.listdir(img_dir):
            exit('error')
    # images1 = images[concept1.replace(' ', '_')]
    # images2 = images[concept2.replace(' ', '_')]
    for prompt in prompts:
        d = {'prompt': prompt,"concept1": concept1, "concept2": concept2}
        data.append(d)
def gen():
    for d in data:
        yield d
ds = Dataset.from_generator(gen)
ds.push_to_hub('ImagenHub/Multi_Subject_Driven_Image_Generation')

to_write_csv = open('/shared/xingyu/projects/custom-diffusion/data/metadata.csv', 'w')
to_write_csv.write('file_name,concept\n')
images = {}
img_dir = '/shared/xingyu/projects/custom-diffusion/data'
for concept in os.listdir(img_dir):
    if '.txt' in concept or '.csv' in concept: continue
    images[concept] = []
    for image in os.listdir(os.path.join(img_dir, concept)):
        to_write_csv.write(os.path.join(concept, image) + ',' + concept.replace('_', ' ') + '\n')

dataset = load_dataset("imagefolder", data_dir="/shared/xingyu/projects/custom-diffusion/data")
dataset.push_to_hub('ImagenHub/Multi_Subject_Concepts')
