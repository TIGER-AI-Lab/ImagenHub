import sys
sys.path.append("src")

import argparse
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from functools import partialmethod
from imagen_hub.benchmark import benchmark_infer, \
                                    infer_control_guided_ig_bench, \
                                    infer_text_guided_ig_bench, \
                                    infer_text_guided_ie_bench, \
                                    infer_mask_guided_ie_bench, \
                                    infer_subject_driven_ig_bench, \
                                    infer_subject_driven_ie_bench, \
                                    infer_multi_concept_ic_bench
try:
    import inquirer
except:
    print("Please install inquirer package to use the interactive mode.")
    print("pip install inquirer")

def parser():
    parser = argparse.ArgumentParser(
        description="benchmarking.py: Running Benchmark scripts for experiment.")
    parser.add_argument("-cfg", "--cfg", type=str,
                        help="Path to the YAML configuration file")
    parser.add_argument("-quiet", "--quiet", action='store_true',
                        help="Disable tqdm progress bar.")
    return parser.parse_args()

def check_arguments_errors(args):
    if args.cfg and not os.path.isfile(args.cfg):
        raise (ValueError("Invalid path {}".format(os.path.abspath(args.cfg))))

def list_config_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.yaml') or f.endswith('.yml')]

def select_config_file():
    config_folder = "benchmark_cfg"
    config_files = list_config_files(config_folder)

    questions = [
        inquirer.List('cfg', message="Select the configuration file", choices=config_files)
    ]
    
    answers = inquirer.prompt(questions)
    return os.path.join(config_folder, answers['cfg'])

def main():
    args = parser()
    check_arguments_errors(args)

    if not args.cfg:
        args.cfg = select_config_file()
    
    config = OmegaConf.load(args.cfg)
    print("=====> Config content:")
    print(OmegaConf.to_yaml(config))
    print("======================")

    # Make tqdm disable
    if args.quiet:
        print("Disabled tqdm.")
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    # Access specific values
    result_folder = config.params.save_to_folder
    limit_images_amount=config.params.limit_images_amount
    experiment_basename = config.params.experiment_basename
    model_list = config.info.running_models
    task_id = config.info.task_id
    if task_id == 0:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_text_guided_ie_bench)
    elif task_id == 1:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_mask_guided_ie_bench)
    elif task_id == 2:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_control_guided_ig_bench)
    elif task_id == 3:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_subject_driven_ie_bench)
    elif task_id == 4:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_multi_concept_ic_bench)
    elif task_id == 5:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_subject_driven_ig_bench)
    elif task_id == 6:
        benchmark_infer(experiment_basename, 
                        model_list = model_list,
                        limit_images_amount = limit_images_amount,
                        result_folder = result_folder,
                        infer_dataset_fn=infer_text_guided_ig_bench)
    else:
        # Implement your new task here
        raise NotImplementedError()


if __name__ == "__main__":
    main()
