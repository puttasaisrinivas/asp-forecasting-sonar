import pandas as pd
import logging
from utils.helpers import convert_to_date, convert_to_quarter
from processing.process_utils import (
    identify_jcodes_constant_forecast,
    identify_jcodes_with_highdrop,
    define_start_end_point,
    impute_constant_drop,
    identify_jcodes_high_growth,
    impute_high_growth,
)


def imputation_constant_high_drop(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    train_qrtr: list,
    forecast_qrtr: list,
) -> pd.DataFrame:
    """
    Perform imputation for constant drop forecasts.

    Args:
        df_actual_train_val (pd.DataFrame): DataFrame containing actual
        training/validation data.
        df_model_forecast (pd.DataFrame): DataFrame containing model forecasts.
        train_qrtr (list): List of training/validation quarters.
        forecast_qrtr (list): List of forecast quarters.

    Returns:
        pandas.DataFrame: DataFrame containing imputed forecasts for constant drop.
    """

    # Identify J_CODES with constant forecast for 8 quarters and
    # first quarter forecast same as last quarter actual
    j_lastQ_actual_copied = identify_jcodes_constant_forecast(
        df_actual_train_val, df_model_forecast, train_qrtr, forecast_qrtr
    )

    # Identify J_CODES with a significant drop (more than 20%) in forecast
    j_high_drop = identify_jcodes_with_highdrop(df_model_forecast, forecast_qrtr)

    # Define start and end points for forecasting
    df_start_end_all = define_start_end_point(
        df_actual_train_val,
        df_model_forecast,
        j_lastQ_actual_copied,
        j_high_drop,
        train_qrtr,
        forecast_qrtr,
    )

    # Impute constant drop forecasts
    return impute_constant_drop(
        df_start_end_all,
        df_model_forecast,
        j_lastQ_actual_copied,
        j_high_drop,
        train_qrtr,
        forecast_qrtr,
    )


def imputation_high_growth(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    train_qrtr: list,
    forecast_qrtr: list,
    j_lastQ_actual_copied: list,
) -> pd.DataFrame:
    """
    Perform imputation for high growth forecasts.

    Args:
        df_actual_train_val (pandas.DataFrame): DataFrame containing
        actual training/validation data.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        train_qrtr (list): List of training/validation quarters.
        forecast_qrtr (list): List of forecast quarters.
        j_lastQ_actual_copied (list): List of J_CODES with last quarter actuals copied.

    Returns:
        pandas.DataFrame: DataFrame containing imputed forecasts for high growth.
    """

    # Identify J_CODES with high forecast growth
    j_high_growth = identify_jcodes_high_growth(
        df_model_forecast, forecast_qrtr, j_lastQ_actual_copied
    )

    # Impute high growth forecasts
    return impute_high_growth(
        df_actual_train_val, df_model_forecast, j_high_growth, train_qrtr, forecast_qrtr
    )


def impute_negative_forecast(
    df_model_forecast: pd.DataFrame,
    df_actual_train_val: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Impute negative forecast values in the dataframe based on certain rules.

    Args:
        df_model_forecast (pandas.DataFrame): DataFrame containing forecasted values.
        df_actual_train_val (pd.DataFrame): DataFrame containing
        actual values used for imputation.
        logger (logging.Logger): Logger to record information and potential issues.

    Returns:
        pandas.DataFrame: DataFrame with imputed negative forecast values.
    """

    child_logger = logger.getChild("IMPUTE_NEGATIVE_FORECAST")
    child_logger.info(
        "Model Forecast dataframe shape:{}".format(df_model_forecast.shape)
    )
    # List of columns to check for negative forecast values
    col_list = ["ASP_PRED_SCALED"]

    df_model_forecast.reset_index(inplace=True)

    # Loop through each column in the list
    for impute_column in col_list:
        # convert into float64
        df_model_forecast[impute_column] = df_model_forecast[impute_column].astype(
            "float64"
        )
        # Find rows with negative forecast values
        neg_preds = df_model_forecast[df_model_forecast[impute_column] < 0]

        # If negative forecast values exist
        if len(neg_preds) > 0:
            # Sort the dataframe
            df_model_forecast = df_model_forecast.sort_values(by=["J_CODE", "QUARTER"])
            df_model_forecast = df_model_forecast.reset_index(drop=True)
            child_logger.info(df_model_forecast.head())

            # Replace negative vals with the latest non-negative value within each group
            gby_series = df_model_forecast.groupby("J_CODE")[impute_column].apply(
                lambda x: x.where(x.gt(0)).ffill(downcast="int")
            )
            child_logger.info("##########################")
            child_logger.info(gby_series)
            df_model_forecast[impute_column] = gby_series.values.tolist()

        # Check for groups with all null forecast values
        null_forecast_groups = df_model_forecast.groupby("J_CODE")[impute_column].apply(
            lambda x: x.isnull().all()
        )
        jcodes_with_all_null_forecasts = null_forecast_groups[
            null_forecast_groups
        ].index.tolist()

        # If there are groups with all null forecasts
        if len(jcodes_with_all_null_forecasts) > 0:
            # Separate df into two:
            # 1) one with non-null forecasts and one with all null forecasts
            # 2) Dataframe with no all null forecasts
            df_model_forecast_without_nulls = df_model_forecast[
                ~df_model_forecast["J_CODE"].isin(jcodes_with_all_null_forecasts)
            ]
            # Dataframe with all null forecasts
            df_model_forecast_with_nulls = df_model_forecast[
                df_model_forecast["J_CODE"].isin(jcodes_with_all_null_forecasts)
            ]

            new_df_without_nulls = pd.DataFrame(
                columns=list(df_model_forecast_without_nulls.columns)
            )

            # Iterate over groups with all null forecasts
            for jcode in jcodes_with_all_null_forecasts:
                # Get actual values for the group
                jcode_df = df_actual_train_val[df_actual_train_val["J_CODE"] == jcode]
                sorted_df = jcode_df.sort_values(by="QUARTER", ascending=False)

                # Replace all null forecast values with the latest actual value for the group
                jcode_forecast_df = df_model_forecast_with_nulls[
                    df_model_forecast_with_nulls["J_CODE"] == jcode
                ]
                # Get the latest actual value
                latest_asp_value = (
                    sorted_df["ASP_TRUE"].dropna().iloc[0]
                    if not sorted_df["ASP_TRUE"].isnull().all()
                    else None
                )
                jcode_forecast_df[impute_column] = latest_asp_value

                # Append the imputed forecasts to the new dataframe
                new_df_without_nulls = pd.concat(
                    [new_df_without_nulls, jcode_forecast_df], axis=0, ignore_index=True
                )

            # Combine dataframes with non-null and imputed forecasts
            child_logger.info(
                "Combining Dataframes with Non-Null and Imputed Forecasts"
            )
            df_model_forecast = pd.concat(
                [df_model_forecast_without_nulls, new_df_without_nulls], axis=0
            )

    return df_model_forecast


def create_smoothened_forecast(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Create a smoothened forecast by imputing constant
    high drop and high growth forecasts.

    Args:
        df_actual_train_val (pandas.DataFrame): DataFrame containing actual training/validation data.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        logger (logging.Logger): Logger to record information and potential issues.

    Returns:
        pandas.DataFrame: DataFrame containing the smoothened forecast.
    """

    child_logger = logger.getChild("CREATE_SMOOTHENED_FORECAST")
    # Identify unique quarters in training and forecast data
    train_qrtr = list(
        df_actual_train_val[["QUARTER"]]
        .drop_duplicates()
        .sort_values(by="QUARTER")
        .values.reshape(-1)
    )
    forecast_qrtr = list(
        df_model_forecast[["QUARTER"]]
        .drop_duplicates()
        .sort_values(by="QUARTER")
        .values.reshape(-1)
    )

    # Impute constant high drop forecasts
    child_logger.info("Imputing Constant High Drop forecasts")
    df_imputed_1 = imputation_constant_high_drop(
        df_actual_train_val, df_model_forecast, train_qrtr, forecast_qrtr
    )

    # Identify J_CODES with last quarter actuals copied
    j_lastQ_actual_copied = list(df_imputed_1.index)

    # Impute high growth forecasts
    child_logger.info("Imputing High Growth")
    df_imputed_2 = imputation_high_growth(
        df_actual_train_val,
        df_model_forecast,
        train_qrtr,
        forecast_qrtr,
        j_lastQ_actual_copied,
    )

    # Concatenate imputed forecasts
    df_output_imputation = pd.concat([df_imputed_1, df_imputed_2])

    # Reshape the dataframe
    df_output_imputation = pd.melt(df_output_imputation.reset_index(), id_vars="index")
    df_output_imputation.columns = ["J_CODE", "QUARTER", "ASP_FORECAST_ADJUSTED"]
    df_output_imputation = df_output_imputation.sort_values(["J_CODE", "QUARTER"])

    # Extract unimputed forecasts from the model forecast data
    df_model_unimputed = df_model_forecast[
        ~df_model_forecast["J_CODE"].isin(df_output_imputation["J_CODE"].unique())
    ][["J_CODE", "QUARTER", "ASP_PRED"]]
    df_model_unimputed.columns = ["J_CODE", "QUARTER", "ASP_PRED"]

    # Rename columns for imputed forecasts
    df_model_imputed = df_output_imputation
    df_model_imputed.columns = ["J_CODE", "QUARTER", "ASP_PRED"]

    # Concatenate unimputed and imputed forecasts
    child_logger.info("Concatenating imputed forecasts")
    return pd.concat([df_model_unimputed, df_model_imputed])

def post_process(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    df_market_event: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Perform post-processing on the model forecasts.

    Args:
        df_actual_train_val (pandas.DataFrame): DataFrame containing actual training/validation data.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        df_market_event (pandas.DataFrame): DataFrame containing market event data.
        logger (logging.Logger): Logger to record information and potential issues.

    Returns:
        pandas.DataFrame: DataFrame containing post-processed forecasts.
    """

    child_logger = logger.getChild("POSTPROCESS")
    df_actual_train_val["QUARTER"] = df_actual_train_val["QUARTER"].astype(str)
    df_model_forecast["QUARTER"] = df_model_forecast["QUARTER"].astype(str)
    # Remove actual training/validation data that overlaps with forecast data
    df_actual_train_val = df_actual_train_val[
        ~df_actual_train_val["QUARTER"].isin(df_model_forecast["QUARTER"])
    ]

    # Impute negative forecasts

    child_logger.info("Imputing Negative Forecast")
    df_model_forecast = impute_negative_forecast(
        df_model_forecast, df_actual_train_val, child_logger
    )

    # Smoothen forecasts
    # To handle the sudden drop/rise in the forecast, after imputing
    child_logger.info("Creating Smootened Forecast - Scaled dataset")

    # Create smoothened forecasts for scaled dataset
    # To fit in the 'create_smoothened_forecast' function need to rename the column 'ASP_pred_scaled' to 'ASP_pred'.
    df_smoothened_forecast_scaled = create_smoothened_forecast(
        df_actual_train_val,
        df_model_forecast.rename({"ASP_PRED_SCALED": "ASP_PRED"}, axis=1),
        child_logger,
    )
    # Reverting the name of the column 'ASP_PRED' to 'ASP_PRED_SCALED'
    df_smoothened_forecast_scaled = df_smoothened_forecast_scaled.rename(
        {"ASP_PRED": "ASP_PRED_SCALED"}, axis=1
    )

    return df_smoothened_forecast_scaled
