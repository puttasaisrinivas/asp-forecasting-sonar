import logging
from utils import constants
from datetime import datetime


def setup_logger(filename: str, files_location: str, name: str) -> logging.Logger:
    """
    Args:
        filename (str): Filename to create .log file
        files_location (str):  path of the files
        name (str):
    Returns:
        logging.Logger -> Object to stream logs to console & File
        filename (str) -> name of the file
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create a full path
    FOLDERS_PATH = files_location
    filename = filename + "_{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    LOG_PATH = "{folder_path}{log_dir}{fname}".format(
        folder_path=FOLDERS_PATH, log_dir=constants.LOGS_FOLDER, fname=filename
    )
    print("log_path:", LOG_PATH)
    # create a file handler and set the level to DEBUG
    fh = logging.FileHandler(filename=LOG_PATH)
    fh.setLevel(logging.DEBUG)

    # create stream handler and set the level to INFO
    # (only displays INFO logs on the console)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # create a formatter and set it for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # add the handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, filename
