# Quickstart

## Benchmarking
To reproduce our experiment reported in the paper:

* First run `benchmarking.py`. The output should be saved into `results` folder (or anywhere you specified in the yaml file.)
```shell
python3 benchmarking.py --help
usage: benchmarking.py [-h] -cfg CFG [-quiet]

benchmarking.py: Running Benchmark scripts for experiment.

optional arguments:
  -h, --help           show this help message and exit
  -cfg CFG, --cfg CFG  Path to the YAML configuration file
  -quiet, --quiet      Disable tqdm progress bar.
```

The yaml configuration file would look something like this:
```yaml
# task_id :
# 0 for Text-Guided Image Editing
# 1 for Mask-Guided Image Editing
# 2 for Control-Guided Image Generation
# 3 for Subject-Driven Image Editing
# 4 for Multi-Concept Image Composition
# 5 for Subject-Driven Image Generation
# 6 for Text-Guided Image Generation

info:
  task_id: 6 # id to determine which benchmark to run for now
  running_models: [
      "SD",
      "SDXL",
      "OpenJourney",
      "DeepFloydIF",
      "DALLE",
      "Kandinsky"
    ] # Determine which model to run
params:
  limit_images_amount: null # Run only certain amount of images. null means running all of them
  save_to_folder: "results"
  experiment_basename: "ImagenHub_Text-Guided_IG" # outputs will be saved to <save_to_folder>/<experiment_basename>/
visualize:
  prioritize_folders: ["DALLE","DeepFloydIF","OpenJourney","SD","SDXL"] # The folder that you want to put in front. Support list.
```
You can freely pick the models that you want to run.

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

* Then after running the experiment, you can run `visualize.py` to produce a `index.html` file for visualization.

```shell
python3 visualize.py --help
usage: visualize.py [-h] -cfg CFG [-width IMG_WIDTH]

visualizer.py: Generating html file according to experiment result.

optional arguments:
  -h, --help            show this help message and exit
  -cfg CFG, --cfg CFG   Path to the YAML configuration file
  -width IMG_WIDTH, --img_width IMG_WIDTH
                        Each image width.
```
The file would look like something like this. We hosted our experiment results on [Imagen Museum](https://chromaica.github.io/#imagen-museum).
<img src="https://i.imgur.com/0uOMhtT.png" width="512" />
We have provided a few yml config for different tasks.

### Text-guided Image Generation
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_t2i.yml
python3 visualize.py --cfg benchmark_cfg/ih_t2i.yml
```

### Mask-guided Image Editing 
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_mask-guided.yml
python3 visualize.py --cfg benchmark_cfg/ih_mask-guided.yml
```

### Text-guided Image Editing
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_text-guided.yml
python3 visualize.py --cfg benchmark_cfg/ih_text-guided.yml
```

### Subject-driven Image Generation
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_sub-gen.yml
python3 visualize.py --cfg benchmark_cfg/ih_sub-gen.yml
```

### Subject-driven Image Editing
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_sub-edit.yml
python3 visualize.py --cfg benchmark_cfg/ih_sub-edit.yml
```

### Multi-concept Image Composition
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_t2i.yml
python3 visualize.py --cfg benchmark_cfg/ih_t2i.yml
```

### Control-guided Image Generation
```shell
python3 benchmarking.py -cfg benchmark_cfg/ih_control-guided.yml
python3 visualize.py --cfg benchmark_cfg/ih_control-guided.yml
```

## Running Metrics
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


## Running single model
```python
import imagen_hub

model = imagen_hub.load("SDXL")
image = model.infer_one_image(prompt="people reading pictures in a museum, watercolor", seed=1)
image
```
<img src="https://i.imgur.com/ruU0BJ0.jpg" width="256" />
