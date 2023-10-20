from .html import HTML
from .file_helper import get_list_in_folder, filter_folders, filter_images
import argparse
import os
from omegaconf import OmegaConf
from typing import Union, List
from datetime import datetime
import json


def gather_outputs(result_folder_path: Union[str, os.PathLike], look_up_images: List = None, input_folder_name: Union[str, list] = 'input'):
    """
    Gathers the outputs of multiple models for building HTML visualization.

    Args:
        result_folder_path (Union[str, os.PathLike]): Path to the root folder containing experiment results. The structure of the folder is expected as mentioned in the function comment.
        look_up_images (List, optional): List of images to look up. If not provided, defaults to None.
        input_folder_name (Union[str, list], optional): Name of the input folder. Defaults to 'input'.

    Returns:
        list: A list with 4 nested lists for each lookup image, which contains the images, texts, links and information.

    Note:
        This function expects a specific structure for the result_folder_path. If the structure is not adhered to, unexpected behavior might occur.
        
    """
    #   Sphinx says unexpected unindent
    #
    #    Expected Folder Structure:
    #    result_root_folder
    #    └── result_folder (your experiment result folder)
    #        ├── input
    #        │   └── image_1.jpg ...
    #        ├── model1
    #        │   └── image_1.jpg ...
    #        ├── model2
    #        │   └── image_1.jpg ...
    #        ├── ...

    json_object = None
    try:
        with open(os.path.join(result_folder_path, 'dataset_lookup.json'), 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
    except:
        print(
            f"=====> json_object {(os.path.join(result_folder_path, 'dataset_lookup.json'))} cannot be loaded.")

    model_names = get_list_in_folder(result_folder_path)
    # filter out anything that is not folder
    model_names = filter_folders(model_names, parent_path=result_folder_path)

    # move 'input' to the front of the list
    if isinstance(input_folder_name, str):
        input_folder_name = [input_folder_name]
    for input_folder in input_folder_name[::-1]: #reverse the list
        if input_folder in model_names:
            model_names.remove(input_folder)
            model_names.insert(0, input_folder)
        else:
            raise FileNotFoundError("We need the input folder for names look up.")

    input_folder_name = input_folder_name[0] # Use the input as lookup. Should be the first one
    if look_up_images == None:
        look_up_images = get_list_in_folder(
            os.path.join(result_folder_path, input_folder_name))
        # filter out anything that is not image
        look_up_images = filter_images(look_up_images, parent_path=os.path.join(
            result_folder_path, input_folder_name))

    gathered_outputs = []
    for look_up_image in look_up_images:
        ims, txts = [], []
        for model_name in model_names:
            model_folder_path = os.path.join(result_folder_path, model_name)
            image_names = get_list_in_folder(model_folder_path)
            if look_up_image in image_names:
                ims.append(os.path.join(model_name, look_up_image))
                txts.append(model_name)
        links = ims  # Links are basically image links
        if len(ims):  # if ims is not empty
            if json_object is not None:
                try:
                    look_up_image = look_up_image + '\n' + \
                        json.dumps(json_object[look_up_image], indent=4)
                except:
                    continue # If there is any extra image
            # Storing [ims], [txts], [links] for each look_up_image
            gathered_outputs.append([ims, txts, links
                                        , look_up_image])
    return gathered_outputs


def build_html(result_folder_path: Union[str, os.PathLike], image_width: int = 512, input_folder_name: Union[str, list] = 'input'):
    """
    Generates an HTML report for visualizing experiment results.

    Args:
        result_folder_path (Union[str, os.PathLike]): Path to the folder containing experiment results.
        image_width (int, optional): Desired width of the images in the HTML report. Defaults to 512.
        input_folder_name (Union[str, list], optional): Name of the input folder. Defaults to 'input'.

    Returns:
        None

    Note:
        The generated HTML report provides an overview of the experiment and its results.
    """
    experiment_basename = str(os.path.basename(result_folder_path))
    html = HTML(web_dir=result_folder_path,
                title=f'Visualize {experiment_basename}')
    html.add_header(f"Visualize Experiment: {experiment_basename}")
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    html.add_header(f"Generated on {dt_string}")
    html.add_header("Images might take a while to load.", header_type=4)
    outputs = gather_outputs(result_folder_path, input_folder_name=input_folder_name)
    for (ims, txts, links, info) in outputs:
        html.add_paragraph(info)
        html.add_images(ims, txts, links, width=image_width)
    html.save()


if __name__ == "__main__":
    def parser():
        parser = argparse.ArgumentParser(
            description="visualizer.py: Generating html file according to experiment result.")
        parser.add_argument("-cfg", "--cfg", required=True, type=str,
                            help="Path to the YAML configuration file")
        return parser.parse_args()


    def check_arguments_errors(args):
        if not os.path.isfile(args.cfg):
            raise (ValueError("Invalid path {}".format(os.path.abspath(args.cfg))))

    args = parser()
    check_arguments_errors(args)
    config = OmegaConf.load(args.cfg)

    # Building html
    build_html(os.path.join(config.params.save_to_folder, config.params.experiment_basename),
               image_width=config.visualize.image_width,
               input_folder_name=config.visualize.prioritize_folders)
    print(
        f"Built index.html in folder {os.path.join(config.params.save_to_folder, config.params.experiment_basename)}.")
