import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import warnings
import logging

# Ignore all warnings to prevent unnecessary output clutter
warnings.filterwarnings("ignore")


def find_best_forecasting_model_autoarima(
    pivot_data: pd.DataFrame,
    product_id: str,
    forecast_months: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Find the best forecasting model using auto_arima and generate predictions.

    Args:
        pivot_data (pd.DataFrame): Pivot table data with 'ASP_MTH' as index
        and product IDs as columns.
        product_id (str): ID of the product for which to generate forecasts.
        forecast_months (int): Number of months to forecast.
        logger (logging.Logger): Logger to log progress and outcomes
        during the function execution.

    Returns:
        pd.DataFrame: DataFrame containing actual and predicted ASP values,
        along with residuals.

    Raises:
        ValueError: If the product_id is not found in the pivot_data
        or if the forecast_months is non-positive.

    Notes:
        - The function uses auto_arima to automatically select
        the best ARIMA model for forecasting.
        - It generates predictions for the specified number of forecast months.
    """

    logger.getChild("BEST_MODEL_AUTOARIMA")
    arima_data = pivot_data.copy()
    arima_data_actual = arima_data.copy()
    arima_data.iloc[-forecast_months:, 1:] = np.NaN

    arima_data = arima_data.set_index("ASP_MTH")
    arima_data_actual = arima_data_actual.set_index("ASP_MTH")

    asp_values = arima_data[product_id]
    asp_values_actual = arima_data_actual[product_id]

    total_months = len(asp_values)
    train_months_len = total_months - forecast_months
    train_data = asp_values[:train_months_len].dropna()

    # Set random seed for reproducibility
    random_seed = 42

    # Fit auto_arima model
    model = auto_arima(
        train_data,
        seasonal=True,
        m=1,
        stepwise=True,
        trace=False,
        error_action="ignore",
        random_state=random_seed,  # Add random_state for reproducibility
    )

    # Generate forecasts
    forecast_train = model.predict_in_sample(return_conf_int=False)
    forecast, conf_int = model.predict(n_periods=forecast_months, return_conf_int=True)
    forecast_all = pd.concat([forecast_train, forecast])
    forecast_all.index = pd.to_datetime(forecast_all.index)

    # Merge actual and predicted ASP values
    arima_forecast_op_data = pd.merge(
        pd.DataFrame(
            {"ASP_MTH": asp_values_actual.index, "ACTUAL_ASP": asp_values_actual.values}
        ),
        pd.DataFrame(
            {"ASP_MTH": forecast_all.index, "ARIMA_PREDICTED_ASP": forecast_all.values}
        ),
        on="ASP_MTH",
        how="left",
    )
    arima_forecast_op_data["J_CODE"] = product_id

    # Arrange columns and calculate residuals
    arima_forecast_op_data = arima_forecast_op_data[
        ["J_CODE", "ASP_MTH", "ACTUAL_ASP", "ARIMA_PREDICTED_ASP"]
    ]
    arima_forecast_op_data["J_CODE"] = arima_forecast_op_data["J_CODE"].astype(str)
    arima_forecast_op_data["ASP_MTH"] = pd.to_datetime(
        arima_forecast_op_data["ASP_MTH"]
    )
    arima_forecast_op_data = arima_forecast_op_data.sort_values(
        by=["J_CODE", "ASP_MTH"]
    )
    arima_forecast_op_data = arima_forecast_op_data.reset_index(drop=True)
    arima_forecast_op_data["RESIDUAL"] = (
        arima_forecast_op_data["ACTUAL_ASP"]
        - arima_forecast_op_data["ARIMA_PREDICTED_ASP"]
    )

    return arima_forecast_op_data


def train_and_predict_with_arima(
    processed_master_data: pd.DataFrame,
    forecast_months: int,
    target_variable: str,
    time_series_columns: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Train and predict with ARIMA model for each product in the processed_master_data.

    Args:
        processed_master_data (pd.DataFrame): Processed master data with
        columns including 'ASP_MTH', 'J_CODE', and 'ASP_TRUE'.
        forecast_months (int): Number of months to forecast.
        target_variable (str): The target variable to be used for training ARIMA model.
        time_series_columns (list): List of columns representing time series data.
        logger (logging.Logger): Logger to log progress and outcomes
        during the function execution.

    Returns:
        pd.DataFrame: DataFrame containing actual ASP values,
        ARIMA predicted ASP values, and residuals.

    Notes:
        - The function trains an ARIMA model for each product
        based on the provided target variable.
        - It generates predictions for the specified number of forecast months.
    """

    # Create a child logger for specific logging within this function
    child_logger = logger.getChild("TRAIN_AND_PREDICT_WITH_ARIMA")

    # Pivot the processed master data
    processed_master_data_pivot_data = pd.pivot(
        data=processed_master_data,
        index="ASP_MTH",
        columns="J_CODE",
        values=target_variable,
    ).reset_index()
    processed_master_data_pivot_data.sort_values(by=["ASP_MTH"], inplace=True)

    forecast_results = []
    # Iterate over each product ID ; Log progress for every 50 products processed
    for idx, product_id in enumerate(processed_master_data_pivot_data.columns[1:]):
        if idx % 50 == 0 and idx != 0:
            child_logger.info(
                "Completed Finding best Auto ARIMA model for N={} products".format(idx)
            )
        # Find the best forecasting model using auto_arima
        arima_output = find_best_forecasting_model_autoarima(
            processed_master_data_pivot_data.copy(),
            product_id,
            forecast_months,
            child_logger,
        )
        forecast_results.append(arima_output)

    # Concatenate forecast results
    arima_predicted_forecast = pd.concat(forecast_results)
    arima_predicted_forecast["J_CODE"] = arima_predicted_forecast["J_CODE"].astype(str)
    arima_predicted_forecast["ASP_MTH"] = pd.to_datetime(
        arima_predicted_forecast["ASP_MTH"]
    )
    arima_predicted_forecast = arima_predicted_forecast.sort_values(
        by=["J_CODE", "ASP_MTH"]
    )
    arima_predicted_forecast = arima_predicted_forecast.reset_index(drop=True)

    # Merge with actual ASP values from processed_master_data
    processed_master_data["J_CODE"] = (
        processed_master_data["J_CODE"].astype(str).astype("object")
    )
    arima_predicted_forecast = pd.merge(
        arima_predicted_forecast,
        processed_master_data[["J_CODE", "ASP_MTH", "ASP_TRUE"]],
        on=["J_CODE", "ASP_MTH"],
        how="left",
    )

    # Fill missing values in time series columns with forward fill
    arima_predicted_forecast[time_series_columns] = arima_predicted_forecast.groupby(
        ["J_CODE"]
    )[time_series_columns].transform(lambda x: x.fillna(method="ffill"))

    return arima_predicted_forecast
