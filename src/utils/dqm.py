import pandas as pd
import pyspark.pandas as ps
import yaml
from utils.helpers import read_data
import logging


def check_tables(config: yaml, catalog_name: str, logger: logging.Logger):
    """
    Checks for the existence and accessibility of
    specified data files in the configuration.

    Args:
        config (yaml): Configuration object containing file paths.
        catalog_name (str): Name of the catalog.
        logger (logging.Logger): Logger object for logging messages.

    Raises:
        Exception: If any specified data file is missing.
    """
    child_logger = logger.getChild("TABLE ACCESS CHECK")
    child_logger.info(
        "Checking for the existence and accessibility of "
        "specified data files in the configuration"
    )
    data_paths = {
        "master_data": config["DATA"]["master_data"],
        "asp_data": config["DATA"]["asp_data"],
        "ep_data": config["DATA"]["ep_data"],
        "lst_selected_products": config["DATA"]["lst_selected_products"],
        "lst_prioritized_products": config["DATA"]["lst_prioritized_products"],
    }

    # Iterate over data paths and check for missing files
    for data_name, data_path in data_paths.items():
        try:
            if data_path.endswith(".csv"):
                pd.read_csv(data_path)
            else:
                data_path = f"{catalog_name}.{data_path}"
                ps.read_table(data_path).to_pandas()
        except Exception as e:
            raise Exception(f"{data_name} data is missing.") from e


def check_features(config: yaml, catalog_name: str, logger: logging.Logger):
    """
    Validates the presence of specified columns in
    data files and the count of prioritized products.

    Args:
        config (yaml): Configuration object containing data paths and expected column names.
        catalog_name (str): Name of the unity catalog
        logger (logging.Logger): Logger object for logging messages.

    Raises:
        Exception: If any expected column is missing in the data files.
        Exception: If the count of prioritized products is not as expected.
    """

    # Log the start of the validation process for required columns in the data files
    child_logger = logger.getChild("CHECKING FEATURES")
    child_logger.info(
        "Validating the presence of specified columns in "
        "data files and the count of prioritized products."
    )

    # Read the data files, extract the list of columns and retrieve the expected list of master data columns
    data_dict = read_data(config, catalog_name, logger)
    master_data_col_list = data_dict["master_data"].columns.to_list()
    config_master_list = config["MODEL_FEATURES"]["MASTER_DATA_FEATURES"][
        "ORDERED_COLUMNS"
    ]

    # Extract the list of columns from the 'ep_data' dataframe,
    # Retrieve the expected list of ep data columns
    ep_data_col_list = data_dict["ep_data"].columns.to_list()
    config_ep_list = config["MODEL_FEATURES"]["EP_DATA_FEATURES"]["COLUMNS"]

    # Get the number of prioritized products by checking the row count of the 'lst_prioritized_products' dataframe
    count_prioritized_prods = data_dict["lst_prioritized_products"].shape[0]

    for i in config_master_list:
        if i not in master_data_col_list:
            raise Exception("Column: " + str(i) + " is not available in master_data")

    for i in config_ep_list:
        if i not in ep_data_col_list:
            raise Exception("Column: " + str(i) + " is not available in ep_data")

    child_logger.info(
        "Total Number of prioritized codes are: " + str(count_prioritized_prods)
    )


def check_j_codes(config: yaml, catalog_name: str, logger: logging.Logger):
    """
    Checks if all prioritized J codes are present in the master data.

    Args:
        config (yaml): Configuration object containing data paths.
        catalog_name (str): Name of the unity catalog
        logger (logging.Logger): Logger object for logging messages.

    Raises:
        Exception: If not all prioritized J codes are present in the master data.
    """

    child_logger = logger.getChild("J_CODE CHECK")
    child_logger.info(
        "Checking if all prioritized J codes are present in the master data"
    )
    data_dict = read_data(config, catalog_name, logger)
    master_data = data_dict["master_data"]
    prioritized_df = data_dict["lst_prioritized_products"]

    # Check if all j_codes in prioritized_df are present in master data
    all_present = prioritized_df["J_CODE"].isin(master_data["J_CODE"]).all()

    if all_present:
        child_logger.info("All prioritized j_codes are present in master_data")
    else:
        raise Exception("Not all prioritized j_codes are present in master_data")


def check_asp_data(config: yaml, catalog_name: str, logger: logging.Logger):
    """
    Checks if there are any J codes with
    zero ASP price in any quarter in the master data.

    Args:
        config (yaml): Configuration object containing data paths.
        catalog_name (str): Name of the unity catalog.
        logger (logging.Logger): Logger object for logging messages.

    Raises:
        Exception: If any J codes have zero ASP price in some quarters.
    """

    #  Log the start of the check for any J codes with zero ASP price in the master data
    child_logger = logger.getChild("ASP DATA CHECK")
    child_logger.info("Checking if there are any J codes with zero ASP price")
    # Read the necessary data files, extract the 'master_data' dataframe and convert the 'CAL_DT' column in 'master_data' to datetime format
    data_dict = read_data(config, catalog_name, logger)
    master_data = data_dict["master_data"]
    master_data["CAL_DT"] = pd.to_datetime(master_data["CAL_DT"])

    # Group by j_code and quarter
    master_data["QUARTER"] = master_data["CAL_DT"].dt.to_period("Q")

    # Check if asp_price is 0 for any j_code and quarter
    result = (
        master_data.groupby(["J_CODE", "QUARTER"])["ASP_PRC"]
        .apply(lambda x: (x == 0).any())
        .reset_index()
    )
    # Filter the result to get only the rows where asp_price_is_zero is True
    j_codes_with_zero_asp_price = result[result["ASP_PRC"]]

    if not j_codes_with_zero_asp_price.empty:
        child_logger.info(j_codes_with_zero_asp_price[["J_CODE", "QUARTER"]])
        raise Exception("There are j_codes with zero asp_price in some quarters.")

    child_logger.info("No j_codes with zero asp_price found. Continuing...")


def dqm_checks(config: yaml, catalog_name: str, logger: logging.Logger):
    """
    Performs a series of data quality and inference checks.

    Args:
        config (yaml): Configuration object containing data paths and settings.
        catalog_name (str): Name of the unity catalog.
        logger (logging.Logger): Logger object for logging messages.
    """

    # Perform a series of checks for the data and configuration
    check_tables(config, catalog_name, logger)
    check_features(config, catalog_name, logger)
    check_j_codes(config, catalog_name, logger)
    check_asp_data(config, catalog_name, logger)
