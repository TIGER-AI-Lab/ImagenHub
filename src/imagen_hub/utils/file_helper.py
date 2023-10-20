import os
from typing import Union, List, Optional
from urllib.parse import urlparse
import requests
from .image_helper import check_is_image

def download_weights_to_directory(url: str, save_dir: str, filename: str) -> str:
    """
    Download model weights from a given URL and save them to a specified directory.

    If the weights file already exists in the specified directory, the download is skipped.

    Args:
        url (str): The URL from which the model weights are to be downloaded.
        save_dir (str): The directory where the weights are to be saved.
        filename (str): The name of the file to save the weights as.

    Returns:
        str: The path to the saved weights file.

    Raises:
        HTTPError: If there was an issue with the download request.
    """

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # If the weights file already exists, no need to re-download
    if os.path.exists(save_path):
        print(f"Weights already exist at {save_path}. Skipping download.")
        return save_path

    # Otherwise, download the weights
    response = requests.get(url)
    response.raise_for_status()

    with open(save_path, 'wb') as out_file:
        out_file.write(response.content)

    return save_path

def get_file_path(filename: Union[str, os.PathLike], search_from: Union[str, os.PathLike] = "."):
    """
    Search for a file across a directory and return its absolute path.

    Args:
        filename (Union[str, os.PathLike]): The name of the file to search for.
        search_from (Union[str, os.PathLike], optional): The directory from which to start the search. Defaults to ".".

    Returns:
        str: Absolute path to the found file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    for root, dirs, files in os.walk(search_from):
        for name in files:
            if name == filename:
                return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError(filename, "not found.")


def read_key_from_file(path: Union[str, os.PathLike]):
    """
    Read the content of a file as a string. Used for reading api keys.

    Args:
        path (Union[str, os.PathLike]): Path to the file.

    Returns:
        str: Content of the file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    try:
        with open(path) as f:
            content = f.read()
        return content
    except:
        raise FileNotFoundError(path, "not found.")


def get_list_in_folder(folder: Union[str, os.PathLike] = "."):
    """
    Get a sorted list of basenames in a directory.

    Args:
        folder (Union[str, os.PathLike], optional): Path to the directory. Defaults to ".".

    Returns:
        List[str]: List of sorted basenames in the directory.
    """
    try:
        from natsort import natsorted
        return natsorted(os.listdir(folder))
    except:
        return sorted(os.listdir(folder))


def filter_folders(item_list: List, parent_path: Union[str, os.PathLike]):
    """
    Filter out items from a list that aren't directories.

    Args:
        item_list (List): List of items to filter.
        parent_path (Union[str, os.PathLike]): Parent path for items in the list.

    Returns:
        List[str]: List containing only directory names.
    """
    return [item for item in item_list if os.path.isdir(os.path.join(parent_path, item))]


def filter_images(item_list: List, parent_path: Union[str, os.PathLike]):
    """
    Filter out items from a list that aren't images.

    Args:
        item_list (List): List of items to filter.
        parent_path (Union[str, os.PathLike]): Parent path for items in the list.

    Returns:
        List[str]: List containing only image file names.
    """
    return [item for item in item_list if check_is_image(os.path.join(parent_path, item))]


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """
    Download a file from a URL using TorchHub or use an existing file if possible.

    Args:
        url (str): The URL from which to download the file.
        model_dir (str): Directory to save the downloaded file.
        progress (bool, optional): Whether or not to display a progress bar. Defaults to True.
        file_name (Optional[str], optional): Name of the file. If not provided, it's inferred from the URL. Defaults to None.

    Returns:
        str: Path to the downloaded or existing file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


# Function to count files in a directory

def count_files_in_directory(directory):
    """
    Count the number of files and subdirectories in a given directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        tuple: A tuple containing:
            - int: The number of files in the directory.
            - int: The number of subdirectories in the directory.

    Raises:
        Exception: If an error occurs while listing items in the directory.
    """
    try:
        # List all files and directories in the given directory
        items = os.listdir(directory)

        # Initialize counters for files and subdirectories
        file_count = 0
        subdirectory_count = 0

        # Loop through each item in the directory
        for item in items:
            item_path = os.path.join(directory, item)

            # Check if the item is a file
            if os.path.isfile(item_path):
                file_count += 1
            # Check if the item is a subdirectory
            elif os.path.isdir(item_path):
                subdirectory_count += 1

        return file_count, subdirectory_count
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 0, 0

def count_files_in_subdirectories(root_directory):
    """
    Count files and subdirectories in all subdirectories of a given root directory.

    Args:
        root_directory (str): The path to the root directory.

    Returns:
        dict: A dictionary where each key is a directory and the value is another
              dictionary containing:
              - "files": The number of files in the directory.
              - "subdirectories": The number of subdirectories in the directory.

    Raises:
        Exception: If an error occurs while walking through the directory tree.
    """
    try:
        # Initialize a dictionary to store the counts
        counts = {}

        # Walk through the directory tree
        for root, dirs, files in os.walk(root_directory):
            # Count files and subdirectories in the current directory
            file_count, subdirectory_count = count_files_in_directory(root)

            # Store the counts in the dictionary
            counts[root] = {"files": file_count,
                            "subdirectories": subdirectory_count}

        return counts
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}
