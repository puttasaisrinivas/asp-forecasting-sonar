import os
import shutil


def cleanup_dir(config) -> bool:
    """
    Cleans up directories before running the pipeline.

    Args:
        config: Dictionary containing configuration settings, including the base directory for files.

    Returns:
        bool: True if directories are successfully cleaned up.
    """
    # Get the base directory for file cleanup from the configuration
    current_directory = config["DATA"]["files_location"]
    print("Directory to clean up:{}".format(current_directory))

    # List of subdirectories to be cleaned
    directories = [
        "data",
        "forecasts",
        "logs",
        "postprocess",
        "results",
        "models",
        "reports",
    ]
    # check if directory exists, and the clean up
    for directory in directories:
        if os.path.exists(current_directory + directory + "/"):
            try:
                # Recursively delete the directory and its contents
                shutil.rmtree(current_directory + directory + "/")

            except Exception as e:
                print("An Error occured while cleaning up directories:{}".format(e))
                raise Exception("Error ocurred while cleaning up the directories")
    print(" Directories cleaned up!")
    return True


def setup_dir(config) -> bool:
    """
    Setup necessary directory in the environment

    Returns:
        boolean: returns True
    """
    # Get current working directory
    current_directory = config["DATA"]["files_location"]
    print("Current Directory to Setup folders:{}".format(current_directory))

    # If the directory exists and ends with "asp-forecasting/", modify the path
    if current_directory:
        if current_directory.endswith("asp-forecasting/"):
            current_directory = "{}data/".format(current_directory)

    # List of directories to be created within the base directory
    directories = [
        "data",
        "forecasts",
        "logs",
        "postprocess",
        "results",
        "models",
        "reports",
    ]

    # Loop through each directory and create it if it does not already exist
    for dir in directories:
        dir_path = "{}{}".format(current_directory, dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    print("Directories are setup....")
