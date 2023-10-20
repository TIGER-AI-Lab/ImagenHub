import os
import pandas as pd
import numpy as np
import traceback
import json
import math
from functools import partial
import krippendorff as kd

def grab_dataframes(root_dir = '.'):
    """
    Grab dataframes from the subdirectories of a given root directory.

    Args:
        root_dir (str): The root directory to start searching for .tsv files. Defaults to the current directory.

    Returns:
        dict: A dictionary where keys are subdirectory names and values are lists of TSV dataframes.
    """
    # Initialize an empty dictionary to store dataframes
    dataframes_dict = {}

    # List all subdirectories in the root directory
    subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Iterate through each subdirectory
    for subdir in subdirectories:
        subdir_path = os.path.join(root_dir, subdir)

        # Forget about ipynb things
        if subdir == ".ipynb_checkpoints":
            continue
        # Initialize a list for dataframes in this directory
        dir_dataframes = []

        # Walk through the current subdirectory
        for root, _, files in os.walk(subdir_path):
            for file in files:
                # Check if the file has a .tsv extension
                if file.endswith('.tsv'):
                    # Create the full path to the TSV file
                    tsv_path = os.path.join(root, file)

                    # Read the TSV file into a pandas dataframe
                    df = pd.read_csv(tsv_path, sep='\t')

                    # Append the dataframe to the list for this directory
                    dir_dataframes.append(df)

        # Add the list of dataframes to the dictionary with the subdirectory name as the key
        dataframes_dict[subdir] = dir_dataframes

    # Now you have a dictionary where keys are subdirectory names and values are lists of TSV dataframes
    return dataframes_dict


def process(one_df, mapping_dict=None):
    """
    Process a single dataframe to obtain various scores.

    Args:
        one_df (DataFrame): The dataframe to process.
        mapping_dict (dict, optional): A dictionary for mapping cell values. Defaults to None.

    Returns:
        dict: A dictionary containing various scores for the dataframe columns.
    """
    df = one_df.drop(one_df.columns[0],axis=1) # Dropping uid column
    #print(df.columns)
    result_dict = {}

    def get_SC_from_cell(cell, return_cell_index=0, mapping = {0: 0, 0.5: 0.5, 1: 1, 2: 2}):
        #print(cell)
        if isinstance(cell, str):
            cell = eval(cell)
        if mapping is not None:
            cell = [mapping[value] for value in cell]
        assert len(cell) == 2

        return cell[return_cell_index]

    get_PR_from_cell = partial(get_SC_from_cell, return_cell_index=1)


    def get_HumanEval_score_from_cell(cell, mapping = {0: 0, 0.5: 0.5, 1: 1, 2: 2}, method = "geometric_mean", alpha=0.7, beta=0.3):
        if isinstance(cell, str):
            cell = eval(cell)
        if mapping is not None:
            cell = [mapping[value] for value in cell]
        if method == "geometric_mean":
            human_score = cell[0]*cell[1]
            human_score = math.sqrt(human_score)
        else:
            human_score = alpha*cell[0]+beta*cell[1]
        return human_score

    for column in df.columns:
        SC_scores = [get_SC_from_cell(x) for x in df[column]]
        PR_scores = [get_PR_from_cell(x) for x in df[column]]
        HumanEval_scores = [get_HumanEval_score_from_cell(x) for x in df[column]]
        SC_mean = float(sum(SC_scores)) / len(SC_scores)
        PR_mean = float(sum(PR_scores)) / len(PR_scores)
        SC_std = np.std(SC_scores)
        PR_std = np.std(PR_scores)
        result_dict[column] = {'SC': SC_scores,
                               'PR': PR_scores,
                               'SC_avg': SC_mean,
                               'PR_avg': PR_mean,
                               'SC_std': SC_std,
                               'PR_std': PR_std,
                               'HumanEval': HumanEval_scores,
                               'HumanEvalSum': sum(HumanEval_scores),
                               'HumanEvalAvg': sum(HumanEval_scores) / len(HumanEval_scores),
                              }
    return result_dict

def combine_result_dicts(result_dicts_list, apply_average=False):
    """
    Combine multiple result dictionaries into one.

    Args:
        result_dicts_list (list): List of dictionaries to be combined.
        apply_average (bool, optional): If set to True, averages will be applied on the results. Defaults to False.

    Returns:
        dict: Combined result dictionary.
    """
    assert isinstance(result_dicts_list, list)
    assert len(result_dicts_list) > 1
    report_dict = {}
    first_result_dict = result_dicts_list[0]
    result_dicts_list.pop(0) # remove first
    for column, value in first_result_dict.items():
        report_dict[column] = {'SC': [ first_result_dict[column]['SC'] ],
                               'PR': [ first_result_dict[column]['PR'] ],
                               'SC_avg': [ first_result_dict[column]['SC_avg'] ],
                               'PR_avg': [ first_result_dict[column]['PR_avg'] ],
                               'SC_std': [ first_result_dict[column]['SC_std'] ],
                               'PR_std': [ first_result_dict[column]['PR_std'] ],
                               'HumanEval': [ first_result_dict[column]['HumanEval'] ],
                               'HumanEvalSum': [ first_result_dict[column]['HumanEvalSum'] ],
                               'HumanEvalAvg': [ first_result_dict[column]['HumanEvalAvg'] ],
                              }

    for result_dict in result_dicts_list:
        for column, value in result_dict.items():
            report_dict[column]['SC'].append(result_dict[column]['SC'])
            report_dict[column]['PR'].append(result_dict[column]['PR'])
            report_dict[column]['SC_avg'].append(result_dict[column]['SC_avg'])
            report_dict[column]['PR_avg'].append(result_dict[column]['PR_avg'])
            report_dict[column]['SC_std'].append(result_dict[column]['SC_std'])
            report_dict[column]['PR_std'].append(result_dict[column]['PR_std'])
            report_dict[column]['HumanEval'].append(result_dict[column]['HumanEval'])
            report_dict[column]['HumanEvalSum'].append(result_dict[column]['HumanEvalSum'])
            report_dict[column]['HumanEvalAvg'].append(result_dict[column]['HumanEvalAvg'])


    if apply_average:
        for column, value in report_dict.items():
            report_dict[column]['SC_std'] = np.std(report_dict[column]['SC_avg'])
            report_dict[column]['PR_std'] = np.std(report_dict[column]['PR_avg'])
            report_dict[column]['HumanEval_std'] = np.std(report_dict[column]['HumanEvalAvg'])
            report_dict[column]['SC'] = np.average(np.array(report_dict[column]['SC']), axis=0)
            report_dict[column]['PR'] = np.average(np.array(report_dict[column]['PR']), axis=0)
            report_dict[column]['SC_avg'] = np.average(np.array(report_dict[column]['SC_avg']), axis=0)
            report_dict[column]['PR_avg'] = np.average(np.array(report_dict[column]['PR_avg']), axis=0)
            report_dict[column]['HumanEval'] = np.average(np.array(report_dict[column]['HumanEval']), axis=0)
            report_dict[column]['HumanEvalSum'] = np.average(np.array(report_dict[column]['HumanEvalSum']), axis=0)
            report_dict[column]['HumanEvalAvg'] = np.average(np.array(report_dict[column]['HumanEvalAvg']), axis=0)
    return report_dict


def get_final_dicts(unprocessed_dataframes, apply_average=False):
    """
    Get final dictionaries by processing raw dataframes.

    Args:
        unprocessed_dataframes (list): List of unprocessed dataframes.
        apply_average (bool, optional): If set to True, averages will be applied on the results. Defaults to False.

    Returns:
        dict: Dictionary containing final results.
    """
    if len(unprocessed_dataframes) == 1:
        for raw_df in unprocessed_dataframes:
            return process(raw_df)
    result_dicts_list = []
    for raw_df in unprocessed_dataframes:
        result_dicts_list.append(process(raw_df))
    return combine_result_dicts(result_dicts_list, apply_average=apply_average)


def sigfig(number, sigfigs=2, digit_mode=True):
    """
    Convert a number to its significant figure representation.

    Args:
        number (float/list): Number or list of numbers to convert.
        sigfigs (int, optional): Number of significant figures to keep. Defaults to 2.
        digit_mode (bool, optional): If set to True, will use the digit mode for formatting. Defaults to True.

    Returns:
        float/list: Number(s) in their significant figure representation.
    """
    if digit_mode:
        string_mode = '{:#.{sigfigs}f}'
    else:
        string_mode = '{:#.{sigfigs}g}'
    if isinstance(number, list):
        new_numbers = []
        for num in number:
            new_num = string_mode.format(num, sigfigs=sigfigs)
            new_numbers.append(float(new_num))
        return new_numbers
    else:
        return float(string_mode.format(number, sigfigs=sigfigs))

def print_all_results(dataframes, apply_average=False):
    """
    Print all results for the given dataframes.

    Args:
        dataframes (dict): Dictionary containing dataframes to print results for.
        apply_average (bool, optional): If set to True, averages will be applied on the results. Defaults to False.
    """
    out = dataframes
    for task_name in out.keys():
        print("=====================>", task_name)
        result = out[task_name]
        if len(result) >= 1:
            result = get_final_dicts(result, apply_average=apply_average)
            for model_name in result.keys():
                print("==>", model_name, ": ")
                print('====> SC_avg | ', sigfig(result[model_name]['SC_avg']))
                print('====> PR_avg | ', sigfig(result[model_name]['PR_avg']))
                print('====> HumanEvalSum | ', sigfig(result[model_name]['HumanEvalSum']))
                print('====> HumanEvalAvg | ', sigfig(result[model_name]['HumanEvalAvg']))
                if apply_average:
                    print('====> SC_std | ', sigfig(result[model_name]['SC_std']))
                    print('====> PR_std | ', sigfig(result[model_name]['PR_std']))
                    print('====> HumanEval_std | ', sigfig(result[model_name]['HumanEval_std']))
        else:
            print(task_name, "| No human eval results yet")
        print("")

def get_one_model_dict(task_name, model_name):
    """
    Retrieve a specific model's dictionary for a given task.

    Args:
        task_name (str): The name of the task.
        model_name (str): The name of the model.

    Returns:
        dict: Dictionary containing the model's results for the task.
    """
    return get_final_dicts(grab_dataframes()[task_name], apply_average=False)[model_name]

def get_fleiss_kappa(np_scores, method="fleiss"):
    """
    Calculate Fleiss' Kappa for given scores. columns as raters.

    Args:
        np_scores (array): Array of scores to calculate kappa for.
        method (str, optional): Method to use for kappa calculation. Defaults to "fleiss".

    Returns:
        tuple: Tuple containing the aggregated raters and the kappa value.
    """
    from statsmodels.stats import inter_rater as irr
    agg = irr.aggregate_raters(np_scores) # returns a tuple (data, categories)
    kappa = irr.fleiss_kappa(agg[0], method=method)
    return agg, kappa

def print_kappa_result(stats, method="fleiss"):
    """
    Print the kappa result for given stats.

    Args:
        stats (array): Array of statistics to calculate and print kappa for.
        method (str, optional): Method to use for kappa calculation. Defaults to "fleiss".
    """
    stats = np.array(stats)
    stats = stats.T
    agg, kappa = get_fleiss_kappa(stats, method=method)
    print("Kappa | ", sigfig(kappa))

def print_kd_result(stats):
    """
    Print the KD result for given stats.

    Args:
        stats (array): Array of statistics to calculate and print KD for.
    """
    stats = np.array(stats)
    kd_value = kd.alpha(stats, level_of_measurement='ordinal')
    print("Kd | ", sigfig(kd_value))

def print_all_kappa_results(dataframes, attr):
    """
    Print all kappa results for given dataframes.

    Args:
        dataframes (dict): Dictionary containing dataframes to print kappa results for.
        attr (str): Attribute to use for kappa calculation.
    """
    out = dataframes
    for task_name in out.keys():
        print("=====================>", task_name)
        result = out[task_name]
        if len(result) >= 1:
            result = get_final_dicts(result, apply_average=False)
            for model_name in result.keys():
                try:
                    print("==>", model_name, ": ")
                    stats = result[model_name][attr]
                    print_kappa_result(stats)
                    print_kd_result(stats)
                except Exception as e:
                    print(e)
                    continue
        else:
            print(task_name, "| No Kappa results yet")
        print("")
