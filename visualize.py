import os
import argparse
from omegaconf import OmegaConf
from imagen_hub.utils.visualizer import build_html
from imagen_hub.utils.file_helper import count_files_in_subdirectories
import sys
sys.path.append("src")

def parser():
    parser = argparse.ArgumentParser(
        description="visualizer.py: Generating html file according to experiment result.")
    parser.add_argument("-cfg", "--cfg", required=True, type=str,
                        help="Path to the YAML configuration file")
    parser.add_argument("-width", "--img_width", required=False, default=256, type=int,
                        help="Each image width.")
    return parser.parse_args()

def check_arguments_errors(args):
    if not os.path.isfile(args.cfg):
        raise (ValueError("Invalid path {}".format(os.path.abspath(args.cfg))))

def main():
    args = parser()
    check_arguments_errors(args)
    config = OmegaConf.load(args.cfg)
    save_path = os.path.join(config.params.save_to_folder,
                             config.params.experiment_basename)

    # Building html
    build_html(result_folder_path=save_path,
               image_width=args.img_width,
               input_folder_name=config.visualize.prioritize_folders)

    counts = count_files_in_subdirectories(save_path)

    # Display the results
    print("=====> HTML content:")
    for directory, count_data in counts.items():
        print(f"Number of Files in {os.path.basename(directory)}: {count_data['files']}")
        #print(f"Directory: {directory}")
        #print(f"  Number of Files: {count_data['files']}")
        #print(f"  Number of Subdirectories: {count_data['subdirectories']}")
    print("======================")

    print(f"Built index.html in folder {save_path}.")

if __name__ == "__main__":
    main()
