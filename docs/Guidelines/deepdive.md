# Learn By Examples

By exploring the examples, you will have a better understanding what can ImagenHub do, and how you can achieve different things with it.

## Ways to instantiate a model

* We recommend using `init_with_default_params=True` to get the predefined model. However you can also instantiate a model without triggering the `__init__` and fill in the init parameters later.

```python
import imagenhub

initialized_model = imagenhub.get_model(model_name='DiffEdit', init_with_default_params=True)

uninitialized_model = imagenhub.get_model(model_name='DiffEdit', init_with_default_params=False)
initialized_model = uninitialized_model(device="cuda", weight="stabilityai/stable-diffusion-2-1")
```

Alternatively, you can use alias such as
* `load`
* `load_model`

Read [https://imagenhub.readthedocs.io/en/latest/source/imagen_hub.infermodels.html#imagen_hub.infermodels.get_model](https://imagenhub.readthedocs.io/en/latest/source/imagen_hub.infermodels.html#imagen_hub.infermodels.get_model) for more info.

## Inference of model

* Each model contains a function `infer_one_image`. Note that models in the same task usually share same set of parameters. Please refer to the docmentations for more info.

### Inference of single model

```python
import imagen_hub

model = imagen_hub.load("SDXL")
image = model.infer_one_image(prompt="people reading pictures in a museum, watercolor", seed=1)
image
```
<img src="https://i.imgur.com/ruU0BJ0.jpg" width="256" />

Read [https://imagenhub.readthedocs.io/en/latest/source/imagen_hub.infermodels.html#imagen-hub-infermodels-package](https://imagenhub.readthedocs.io/en/latest/source/imagen_hub.infermodels.html#imagen-hub-infermodels-package) for more info.

### Inference of multiple models

```python
import imagen_hub
from imagen_hub.loader.infer_dummy import load_text_guided_ig_data
from imagen_hub.utils import save_pil_image, get_concat_pil_images

dummy_data = load_text_guided_ig_data(get_one_data=True)
print(dummy_data)
prompt = dummy_data['prompt']
model_list = ["SD", "SDXL"]
image_list = []
for model_name in model_list:
    model = imagen_hub.load(model_name)
    output = model.infer_one_image(prompt=prompt, seed=42).resize((512,512))
    image_list.append(output) 

show_image = get_concat_pil_images(image_list)
show_image
```
`{'prompt': 'A black colored banana.', 'category': 'Colors', 'source': 'DrawBench', 'uid': 0}`
<img src="https://i.imgur.com/WAdNS8Q.jpg" width="512" />

## Getting autometrics

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

# You can perform multiple images in the similar manner
def evaluate_all(model, list_real_images, list_generated_images):
  score = [model.evaluate(x, y) for (x,y) in zip(list_real_images, list_generated_images)]
  print("====> Avg Score: ", sum(score) / len(score))

show_image
```
<img src="https://i.imgur.com/af8CB4c.jpg" width="512" />

Beside LPIPS, we also support different common metrics for Gen AI.

## Running custom benchmark using the ImagenHub library

Now you know the basic functionalities of ImagenHub. But it seems you want more customization. In fact, it is easy to set up your own experiment.

What if you want to test the different built-in models with a custom dataset?

### Understand the Core Functions

Our benchmark rely on the function [benchmark_infer](https://imagenhub.readthedocs.io/en/latest/source/imagen_hub.benchmark.html#imagen_hub.benchmark.benchmark_infer).

This function:
* Iterates over a list of models
* Initializes each model
* Calls the `infer_dataset_fn` with the model and other arguments to perform the benchmarking

Before you can use benchmark_infer, you need a function that performs the actual benchmarking given a model and dataset. This is what `infer_dataset_fn` is.

Your benchmarking function should:

* Take a model as its first argument.
* Have named arguments experiment_name, result_folder, and limit_images_amount.
* Perform inference using the provided model.
* Store the results in the specified result folder.
* Dump the file info as 'dataset_lookup.json' and 'dataset_lookup.csv' (If you want to support visualize html output)

For example:

```python
from imagen_hub.loader.benchmark_loader import dump_dataset_info
from imagen_hub.benchmark import benchmark_infer
from datasets import load_dataset
def my_infer_dataset_fn(model, experiment_name, result_folder, limit_images_amount):
    # Your code to perform inference with the model and store the results
    data = load_dataset("ImagenHub/Text_to_Image")['eval']

    def process_dataset_uid(sample):
        uid = sample['uid']
        return f"sample_{uid}.jpg"

    # Saving dataset info to a json file if first time or overwrite_inputs=True
    if overwrite_inputs or not os.path.exists(os.path.join(result_folder, experiment_name, 'dataset_lookup.json')):
        dump_dataset_info(data,
                          required_attributes=['prompt', 'category', 'source'],
                          uid_preprocess_fn=process_dataset_uid,
                          save_folder_path=os.path.join(result_folder, experiment_name),
                          filename_wo_ext='dataset_lookup')

    print("========> Running Benchmark Dataset:", experiment_name, "| Model:", model.__class__.__name__)
    index = 0
    for sample in tqdm(data):
        file_basename = process_dataset_uid(sample)
        dest_folder = os.path.join(result_folder, experiment_name, model.__class__.__name__)
        dest_file = os.path.join(dest_folder, file_basename)
        if overwrite_model_outputs or not os.path.exists(dest_file):
            print("========> Inferencing", dest_file)
            prompt = sample['prompt']
            output = model.infer_one_image(prompt=prompt)
            output = output.resize((512, 512), Image.LANCZOS)

            save_pil_image(output, dest_folder, file_basename)
        else:
            print("========> Skipping", dest_file, ", it already exists")
        index += 1
        if limit_images_amount is not None and (index >= limit_images_amount):
            break

result_folder = "results"
limit_images_amount=None
experiment_basename = "My_Exp"
model_list = ["SD", "SDXL"]
benchmark_infer(experiment_basename, 
                model_list = model_list,
                model_init_with_default_params=True,
                limit_images_amount = limit_images_amount,
                result_folder = result_folder,
                infer_dataset_fn=my_infer_dataset_fn)
```

The code is less than 100 lines.

The `dump_dataset_info` function is a utility for extracting specific attributes from a dataset and saving the extracted data as both a JSON and a CSV file.

* JSON file for generating visualization page
* CSV file for porting excel files for human evaluation.

```python
result_folder = "results"
experiment_name = "My_Exp"
data = load_dataset("ImagenHub/Text_to_Image")['eval']
def process_dataset_uid(sample):
    uid = sample['uid']
    source = sample['source']
    category = sample['category']
    prompt = sample['prompt']
    return f"sample_{uid}.jpg"
dump_dataset_info(data,
                    required_attributes=['prompt', 'category', 'source'], # extracting only the relevant attributes
                    uid_preprocess_fn=process_dataset_uid, # using filename as a uid
                    save_folder_path=os.path.join(result_folder, experiment_name),
                    filename_wo_ext='dataset_lookup' # The base filename for the saved files
                 )
```

The output structure would look like:

```shell
results$ tree -I '*jpg'
.
└── My_Exp
    ├── dataset_lookup.csv
    ├── dataset_lookup.json
    ├── index.html
    ├── SD
    └── SDXL
```

where 
* `dataset_lookup.json` storing the attributes of each inference data.
* `dataset_lookup.csv` storing the filenames.
