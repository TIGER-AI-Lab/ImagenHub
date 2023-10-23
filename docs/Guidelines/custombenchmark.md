# Modify default benchmark

In the given script `benchmarking.py`, we have a structure that allows for easy benchmarking of various machine learning models on different tasks. Here's a step-by-step guide on how to use this framework to write your own benchmark:

```shell
python3 benchmarking.py --help
usage: benchmarking.py [-h] -cfg CFG [-quiet]

benchmarking.py: Running Benchmark scripts for experiment.

optional arguments:
  -h, --help           show this help message and exit
  -cfg CFG, --cfg CFG  Path to the YAML configuration file
  -quiet, --quiet      Disable tqdm progress bar.
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

## You want to test with different built-in models with ImagenHub dataset

As mentioned, the `benchmarking.py` rely on a yaml file to perform benchmarking. The yaml configuration file would look something like this:
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

### Prepare Your Configuration

Your configuration should at least have:

* save_to_folder: Where to store results.
* limit_images_amount: If you want to limit the number of data points/images.
* experiment_basename: The base name for the experiment.
* running_models: A list of models you want to benchmark.
* task_id: An identifier for the benchmarking task.

### Run Your Benchmark

Once everything is set up, you can run the script:

```shell
python benchmarking.py --cfg path_to_your_config.yml
```

Don't forget the generate the visualize html file after running your experiment.

```shell
python visualize.py --cfg path_to_your_config.yml --img_width 256
```