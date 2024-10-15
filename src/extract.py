import logging
import pandas as pd
import yaml
from config import ConfigDetails
from processing.process import (
    process_ep_data,
    process_market_event_features,
    process_master_data,
)


def data_process(
    config: yaml,
    config_obj: ConfigDetails,
    logger: logging.Logger,
    master_data_imputed: pd.DataFrame,
    asp_data: pd.DataFrame,
    forecast_month_start: str,
) -> tuple:
    """
    Process data for forecasting.

    Args:
        config (yaml): Configuration details.
        config_obj (ConfigDetails): Object containing configuration details.
        logger (logging.Logger): Logger object for logging messages.
        master_data_imputed (pd.DataFrame): Imputed master data.
        asp_data (pd.DataFrame): ASP Data from CMS website.
        forecast_month_start (str): start month of forecasting
    Returns:
        tuple: A tuple containing processed dataframes for master data,
        EP data, market event data, actual ASP monthly, and actual ASP quarterly.
    """

    child_logger = logger.getChild("EXTRACTION")
    child_logger.info(" Data Processing started")

    ep_data_processed = process_ep_data(
        ep_data=config_obj.ep_data,
        asp_data=asp_data,
        start_date=forecast_month_start,
        forecast_months=config_obj.forecast_months,
        ep_data_date_cols=config["MODEL_FEATURES"]["EP_DATA_FEATURES"]["DATE_COLUMNS"],
        processed_ep_data_cols=config["MODEL_FEATURES"]["EP_DATA_FEATURES"][
            "PROCESSED_EP_DATA_COLUMNS"
        ],
        available_codes=config["MODEL_FEATURES"]["EP_DATA_FEATURES"]["AVAILABLE_CODES"],
        unavailable_codes=config["MODEL_FEATURES"]["EP_DATA_FEATURES"][
            "UNAVAILABLE_CODES"
        ],
        logger=child_logger,
    )
    # ep_data_processed.to_csv('ep_features.csv',index=False)
    child_logger.info("EP Data Processed Shape:{}".format(ep_data_processed.shape))

    market_event_data_processed = process_market_event_features(ep_data_processed)
    child_logger.info(
        "Market Event Data Processed Shape:{}".format(market_event_data_processed.shape)
    )

    master_data_processed = process_master_data(
        master_data_imputed,
        market_event_data_processed,
        master_data_ordered_cols=config["MODEL_FEATURES"]["MASTER_DATA_FEATURES"][
            "ORDERED_COLUMNS"
        ],
        master_data_date_cols=config["MODEL_FEATURES"]["MASTER_DATA_FEATURES"][
            "DATE_COLUMNS"
        ],
        market_event_cols=config["MODEL_FEATURES"]["MARKET_EVENT_FEATURES"]["COLUMNS"],
    )
    child_logger.info("Master Processed Shape:{}".format(master_data_processed.shape))
    master_data_processed = master_data_processed[
        master_data_processed["J_CODE"].isin(config_obj.lst_available_jcodes)
    ]
    market_event_data_processed = market_event_data_processed[
        market_event_data_processed["J_CODE"].isin(config_obj.lst_available_jcodes)
    ]
    child_logger.info(
        "Master Processed Shape after filtering for selected J_CODES:{}".format(
            master_data_processed.shape
        )
    )
    actual_asp_monthly = master_data_processed[
        master_data_processed["J_CODE"].isin(config_obj.lst_available_jcodes)
    ][["J_CODE", "ASP_MTH", "ASP_TRUE"]].reset_index(drop=True)
    actual_asp_monthly["QUARTER"] = actual_asp_monthly["ASP_MTH"].dt.to_period("Q")
    actual_asp_quarterly = (
        actual_asp_monthly.groupby(["J_CODE", "QUARTER"])
        .agg({"ASP_TRUE": "first"})
        .reset_index()
    )

    child_logger.info("Processed the data successfully!")
    return (
        master_data_processed,
        ep_data_processed,
        market_event_data_processed,
        actual_asp_monthly,
        actual_asp_quarterly,
    )
