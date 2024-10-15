import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import pyspark.pandas as ps


def read_data(configuration: str, catalog_name: str, logger: logging.Logger) -> dict:
    """
    Read data from various files based on the provided configuration.

    Args:
        configuration (str): File path to the configuration file.
        catalog_name (str): Name of the unity catalog to read the tables from.
        logger (logging.Logger): Logger object for logging messages.

    Returns:
        dict: A dictionary containing various dfs and values read from different files.
            - master_data (pd.DataFrame or pa.Table): Master data dataframe containing
            exhaustive features corresponding to each jcode.
            - asp_data (pd.DataFrame or pa.Table): Dataframe containing
            product name and other features at jcode level.
            - ep_data (pd.DataFrame): DataFrame containing external information
            like competitor launch details and patent expiry details.
            - lst_selected_products (pd.DataFrame): df with list of selected products.
            - lst_prioritized_products (pd.DataFrame): df with list of prioritized prds.
            - forecast_months (str): Number of forecast months.

    Note:
        - The file paths in the configuration file should be enclosed in "".
        - The master_data and asp_data can be either pandas DataFrame
        or PyArrow Table, depending on the file format.
    """

    child_logger = logger.getChild("read_data")
    # Extract file paths from the configuration
    master_data_path = configuration["DATA"]["master_data"]
    asp_data_path = configuration["DATA"]["asp_data"]
    ep_data_path = configuration["DATA"]["ep_data"]
    lst_selected_products_path = configuration["DATA"]["lst_selected_products"]
    lst_prioritized_products_path = configuration["DATA"]["lst_prioritized_products"]
    forecast_months = configuration["DATA"]["forecast_months"]

    # Load master data based on the file extension. If it's a CSV, use pandas read_csv.
    # Otherwise, load it using pyspark.pandas (ps).
    if ".csv" in master_data_path:
        master_data = pd.read_csv(master_data_path)
    else:
        master_data_path = f"{catalog_name}.{master_data_path}"
        master_data = ps.read_table(master_data_path).to_pandas()
        master_data = master_data[
            configuration["MODEL_FEATURES"]["MASTER_DATA_FEATURES"]["ORDERED_COLUMNS"]
        ]

    # Load ASP data similarly depending on the file extension.
    if ".csv" in asp_data_path:
        asp_data = pd.read_csv(asp_data_path)
    else:
        asp_data_path = f"{catalog_name}.{asp_data_path}"
        asp_data = ps.read_table(asp_data_path).to_pandas()

    # Load ep_data with the same approach as the previous datasets.
    if ".csv" in ep_data_path:
        ep_data = pd.read_csv(ep_data_path)
    else:
        ep_data_path = f"{catalog_name}.{ep_data_path}"
        ep_data = ps.read_table(ep_data_path).to_pandas()

    # Load the list of selected products, based on whether it's a CSV file or not.
    if ".csv" in lst_selected_products_path:
        lst_selected_products = pd.read_csv(lst_selected_products_path)

    else:
        lst_selected_products_path = f"{catalog_name}.{lst_selected_products_path}"
        lst_selected_products = ps.read_table(lst_selected_products_path).to_pandas()

    # Load the list of prioritized products, 
    # filter only those marked with 'YES' in the PRIORITY_FLAG column.
    if ".csv" in lst_prioritized_products_path:
        lst_prioritized_products = pd.read_csv(lst_prioritized_products_path)
        lst_prioritized_products = lst_prioritized_products[
            lst_prioritized_products["PRIORITY_FLAG"] == "YES"
        ][["J_CODE"]]
    else:
        lst_prioritized_products_path = (
            f"{catalog_name}.{lst_prioritized_products_path}"
        )
        lst_prioritized_products = ps.read_table(
            lst_prioritized_products_path
        ).to_pandas()
        lst_prioritized_products = lst_prioritized_products[
            lst_prioritized_products["PRIORITY_FLAG"] == "YES"
        ][["J_CODE"]]

    # Log the shapes of the loaded dataframes for debugging purposes.
    child_logger.debug("Master dataframe Shape:{}".format(master_data.shape))
    child_logger.debug("EP Dataframe shape:{}".format(ep_data.shape))
    child_logger.debug("ASP dataframe shape:{}".format(asp_data.shape))
    child_logger.debug(
        "Prioritized List dataframe shape:{}".format(lst_prioritized_products.shape)
    )
    child_logger.debug(
        "Selected Product dataframe shape:{}".format(lst_selected_products.shape)
    )

    # Convert forecast_months to int
    forecast_months = int(forecast_months)

    # Create a dictionary containing the read data
    return {
        "master_data": master_data,
        "asp_data": asp_data,
        "ep_data": ep_data,
        "lst_selected_products": lst_selected_products,
        "lst_prioritized_products": lst_prioritized_products,
        "forecast_months": forecast_months,
        "files_location": configuration["DATA"]["files_location"],
    }


def get_config_from_file(filename: str) -> yaml:
    """
    Load configuration data from a YAML file and return a Config object.

    Args:
        filename (str): The path to the YAML file containing configuration data.

    Returns:
        Config (yaml): A Config object initialized with the data from the YAML file.

    Notes:
        - Uses the `yaml.safe_load` method for secure parsing of the YAML file.
        - Returns an empty dictionary if the file has no content.
    """

    # Load the file contents using yaml.safe_load for security reasons
    with open(filename, "r") as f:
        contents = yaml.safe_load(f)
        if contents is None:
            contents = {}
        return contents


def generate_month_year_list(start_date: str, num_months: int) -> List[str]:
    """
    Generate a list of month-year strings starting from the given start date.

    Args:
        start_date (str): Start date in the format 'YYYY-MM'.
        num_months (int): Number of months to generate.

    Returns:
        List[str]: List of month-year strings.

    Raises:
        ValueError: If the start_date format is invalid.

    Notes:
        - Each month-year string is formatted as 'YYYY-MM'.
        - Assumes each month has 30 days for simplicity.
    """

    # Generate month-year list
    current_date = datetime.strptime(start_date, "%Y-%m")
    month_year_list = []
    for _ in range(num_months):
        month_year_list.append(current_date.strftime("%Y-%m"))
        current_date += timedelta(
            days=31
        )  # Assuming each month has 31 days for simplicity

    return month_year_list


def get_config(config: yaml, generic_flag: bool, forecast_months: int) -> dict:
    """
    Generate configuration dictionary based on provided parameters.

    Args:
        config (yaml): A Config object initialized with the data from the YAML file.
        generic_flag (bool): Flag indicating whether the analysis is for gnrc products.
        forecast_months (int): Number of forecast months.

    Returns:
        dict: A dictionary containing configuration parameters for the analysis.

    Notes:
        - For generic products, the target variable is "ASP_growth"
        and static columns include basic product information.
        - For non-generic products, the target variable is "ASP_true"
        and static columns include more detailed product information.
        - The window size for generic products is set to 2,
        while for non-generic products it is set to 24.
    """

    config_dict = {
        "forecast_months": forecast_months,
        "time_series_columns": ["RESIDUAL"],
        "time_series": "RESIDUAL",
        "forecast_for_priotized_products_only": True,
        "time_cap": 36,
    }

    if generic_flag:
        config_dict.update(
            {
                "target_variable": config["MODEL_FEATURES"]["GENERIC_PRODUCT"][
                    "TARGET"
                ],
                "static_columns": config["MODEL_FEATURES"]["GENERIC_PRODUCT"][
                    "STATIC_COLUMNS"
                ],
                "window_size": config["MODEL_FEATURES"]["GENERIC_PRODUCT"][
                    "WINDOW_SIZE"
                ],
            }
        )
    else:
        config_dict.update(
            {
                "target_variable": config["MODEL_FEATURES"]["NON_GENERIC_PRODUCT"][
                    "TARGET"
                ],
                "static_columns": config["MODEL_FEATURES"]["NON_GENERIC_PRODUCT"][
                    "STATIC_COLUMNS"
                ],
                "window_size": config["MODEL_FEATURES"]["NON_GENERIC_PRODUCT"][
                    "WINDOW_SIZE"
                ],
            }
        )

    return config_dict


def convert_to_quarter(date_str: str) -> str:
    """
    Convert a date string to a quarter format (YYYYQX).

    Args:
        date_str (str): The input date string in the format "MM/DD/YYYY".

    Returns:
        str: The quarter format representation of the input date.
    """

    # Convert the input date string to a datetime object
    date_object = datetime.strptime(date_str, "%m/%d/%Y")

    # Extract the year and quarter from the datetime object
    year = date_object.year
    quarter = (date_object.month - 1) // 3 + 1

    # Format the result as "YYYYQX"
    return f"{year}Q{quarter}"


def convert_to_date(quarter_str: str) -> str:
    """
    Convert a quarter format (YYYYQX) to a date string (MM/DD/YYYY).

    Args:
        quarter_str (str): The input quarter format string in the format "YYYYQX".

    Returns:
        str: The date string representation of the input quarter.
    """

    # Extract the year and quarter from the input string
    year, quarter = map(int, quarter_str.split("Q"))

    # Calculate the month corresponding to the start of the quarter
    month = (quarter - 1) * 3 + 1

    # Create a datetime object for the first day of the quarter
    date_object = datetime(year, month, 1)

    # Format the result as "MM/DD/YYYY"
    return date_object.strftime("%m/%d/%Y")
