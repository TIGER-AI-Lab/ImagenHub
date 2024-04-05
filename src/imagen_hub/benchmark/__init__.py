from imagen_hub import infermodels
from typing import Union, Optional, Callable
import traceback

from .control_guided_ig import infer_control_guided_ig_bench
from .mask_guided_ie import infer_mask_guided_ie_bench
from .text_guided_ie import infer_text_guided_ie_bench
from .text_guided_ig import infer_text_guided_ig_bench
from .subject_driven_ie import infer_subject_driven_ie_bench
from .subject_driven_ig import infer_subject_driven_ig_bench
from .multi_concept_ic import infer_multi_concept_ic_bench

def benchmark_infer(experiment_basename: str,
                    infer_dataset_fn: Callable,
                    model_list: list,
                    model_init_with_default_params: bool = True,
                    result_folder="results",
                    limit_images_amount=None,
                    pass_model_as_string=False):
    """
    Benchmark various inference models on a given dataset.

    This function iterates through each model provided in `model_list`, initializes them,
    and runs the provided `infer_dataset_fn` to benchmark them. If any model encounters
    an exception during the benchmarking, it captures and prints the traceback without
    stopping the benchmarking for remaining models.

    Parameters:
        experiment_basename (str): Basename for the experiment.
        infer_dataset_fn (Callable): A function to instantiate the dataset and perform
                                     inference. Users are expected to provide this function
                                     tailored to their specific experiment.
        model_list (list): A list of model names to be retrieved from `infermodels`.
        model_init_with_default_params (bool, optional): Whether to initialize the model
                                                         with default parameters. Defaults to True.
        result_folder (str, optional): Folder to store benchmark results. Defaults to "results".
        limit_images_amount (Optional[int], optional): Limits the number of images to be
                                                       benchmarked, if provided. Defaults to None.
        pass_model_as_string (bool, optional): Whether to pass the model as string instead of instances or object. Defaults to False.

    Returns:
        None

    Notes:
        the infer_dataset_fn must have first parameter as model (str, class instance).
        then it must contain the parameter experiment_name, result_folder, and limit_images_amount.
    """
    for model_name in model_list:
        try:
            if pass_model_as_string:
                infer_dataset_fn(model_name,
                                experiment_name=experiment_basename,
                                result_folder=result_folder,
                                limit_images_amount=limit_images_amount)
            else:
                model = infermodels.get_model(
                    model_name=model_name,
                    init_with_default_params=model_init_with_default_params)
                infer_dataset_fn(model,
                                experiment_name=experiment_basename,
                                result_folder=result_folder,
                                limit_images_amount=limit_images_amount)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"{model_name} | Error: {e}\n{tb}")
            continue
