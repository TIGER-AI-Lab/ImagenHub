# Adding new models

* You developed a new model / framework that perform very good results. Now you want to benchmark it with other models. How you can do it?

In this guide we will be adding new model to the codebase and extend the code.

## Integrating your model into ImagenHub


To add your model codebase into ImagenHub codebase, you must modify the following folders:

* `src/imagen_hub/infermodels` : where you create a class interface for the model inference.
* `src/imagen_hub/pipelines` : where you move your codebase into it without much tidy up work.

### How to write the infermodel class
The infermodel class is designed to have minimal methods. However, it must contain the following methods:

* `__init__(args)` for class initialization.
* `infer_one_image(args)` to produce 1 image output. Please try to set the seed as 42.

In that case, you will add a new file in `infermodels` folder.
`infermodels/awesome_model.py`
```python
import torch
import PIL
from imagen_hub.pipelines.awesome_model import AwesomeModelPipeline
class AwesomeModelClass():
    """
    A wrapper ...
    """
    def __init__(self, device="cuda"):
        """
        docstring
        """
        self.pipe = AwesomeModelPipeline(device=device)

    def infer_one_image(self, prompt, seed=42):
        """
        docstring
        """
        self.pipe.set_seed(seed)
        image = self.pipe(prompt=prompt).images[0]
        return image
```
Then you can add a line in `infermodels/__init__.py`:
```shell
from .awesome_model import AwesomeModelClass
```

### Writing your pipeline
About `AwesomeModelPipeline`, it means you need to write a Pipeline file that wraps the function of your codebase, such that the infermodel class can call it with ease.

We recommend structuring code in the `pipelines` folder in this way:

```shell
└── awesome_model
    ├── pipeline_awesome_model.py
    ├── awesome_model_src
    │   └── ...
    └── __init__.py
```

## Running experiment with new model
After finishing and reinstalling the package through 
```shell
pip install -e .
```

You should be able to use the new model. You can adhere to our ImagenHub benchmark by modifying the benchmark_cfg to add your new model in the yml file.

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
      # ...
      "AwesomeModelClass"
    ] # Determine which model to run
params:
  limit_images_amount: null # Run only certain amount of images. null means running all of them
  save_to_folder: "results"
  experiment_basename: "ImagenHub_Text-Guided_IG" # outputs will be saved to <save_to_folder>/<experiment_basename>/
#...
```

Then run the `benchmarking.py` script and `visualize.py` with the modified yaml file.

### Caution: Extra work for certain model tasks

* Note that you might need to modify the source code in `src/imagen_hub/benchmark/` if your task is:
    * Multi-Concept Image Composition : `benchmark/multi_concept_ic.py`
    * Subject-Driven Imagen Generation : `benchmark/subject_driven_ig.py`
    * Subject-Driven Imagen Editing : `benchmark/subject_driven_ie.py`
This is because different models have different behaviours that cannot be aligned.

### Modifying the benchmark source

* Use `isinstance(model, YourModelClass)` to differentiate the model behaviours.

### Matching environment
Make sure the code can be run with the ImagenHub environment. If new dependency is added, please add them to the env_cfg file.

## Submitting your model as through a PR

Finally, you can submit this new model through submiting a Pull Request! Make sure it match the code style in our contribution guide.