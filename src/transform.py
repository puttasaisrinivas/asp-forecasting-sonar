import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_static_features(
    processed_master_data: pd.DataFrame,
    generic_flag: bool,
    forecast_months: int,
    static_columns: list,
) -> pd.DataFrame:
    """
    Extract static features from the processed master data.

    Args:
        processed_master_data (pd.DataFrame): Processed master data
        containing necessary columns.
        generic_flag (bool): A flag indicating whether the generic
        features should be extracted.
        forecast_months (int): Number of months to forecast.
        static_columns (list): List of static columns to be extracted.

    Returns:
        pd.DataFrame: DataFrame containing static features for each product.

    Notes:
        - The function extracts static features based on the
        provided generic flag and static columns.
        - It filters the data based on the training period
        defined by the forecast months.
        - Static columns are imputed with mode values within each product group.
        - Product hierarchy columns are aggregated and reduced to the top 15 values.
    """

    train_last_month = str(
        processed_master_data["ASP_MTH"].unique()[-forecast_months]
    ).split("T")[0]

    if generic_flag:
        static_feature_data = processed_master_data[
            [
                "J_CODE",
                "ASP_MTH",
                "PROD_NAME",
                "THERA_CLS_DSCR",
                "DISEASE_ST",
                "PROD_HIER1",
                "PROD_HIER2",
                "PROD_HIER3",
            ]
        ]
        static_feature_data = static_feature_data[
            static_feature_data["ASP_MTH"] < train_last_month
        ]
        static_feature_data = static_feature_data[
            [
                "J_CODE",
                "PROD_NAME",
                "THERA_CLS_DSCR",
                "DISEASE_ST",
                "PROD_HIER1",
                "PROD_HIER2",
                "PROD_HIER3",
            ]
        ]
    else:
        static_feature_data = processed_master_data[
            [
                "J_CODE",
                "ASP_MTH",
                "PROD_NAME",
                "THERA_CLS_DSCR",
                "MSPN_DSG_FRM",
                "MSPN_ROA",
                "USC5_DSCR",
                "DISEASE_ST",
                "PROD_HIER1",
                "PROD_HIER2",
                "PROD_HIER3",
            ]
        ]
        static_feature_data = static_feature_data[
            static_feature_data["ASP_MTH"] < train_last_month
        ]
        static_feature_data = static_feature_data[
            [
                "J_CODE",
                "PROD_NAME",
                "THERA_CLS_DSCR",
                "MSPN_DSG_FRM",
                "MSPN_ROA",
                "USC5_DSCR",
                "DISEASE_ST",
                "PROD_HIER1",
                "PROD_HIER2",
                "PROD_HIER3",
            ]
        ]

    for column in static_columns:
        static_feature_data[column] = static_feature_data.groupby("J_CODE")[
            column
        ].transform(lambda x: x.mode().values[0] if len(x.mode()) > 0 else np.NaN)

    static_hierarchy_columns = ["PROD_HIER1", "PROD_HIER2", "PROD_HIER3"]
    static_data_hierarchy = static_feature_data[
        ["J_CODE"] + static_hierarchy_columns
    ].drop_duplicates(subset=["J_CODE"], keep="last")
    static_feature_data["J_CODE"] = (
        static_feature_data["J_CODE"].astype(str).astype("object")
    )
    static_data_hierarchy["J_CODE"] = (
        static_data_hierarchy["J_CODE"].astype(str).astype("object")
    )

    static_feature_data = pd.merge(
        static_feature_data.drop(columns=["PROD_HIER1", "PROD_HIER2", "PROD_HIER3"]),
        static_data_hierarchy,
        how="left",
        on="J_CODE",
    )

    static_feature_data["PROD_HIER1"] = static_feature_data["PROD_HIER1"].where(
        static_feature_data["PROD_HIER1"].isin(
            static_feature_data["PROD_HIER1"].value_counts().head(15).index
        ),
        "others",
    )

    static_feature_data.drop(["PROD_HIER2", "PROD_HIER3"], axis=1, inplace=True)
    return static_feature_data.drop_duplicates().reset_index(drop=True)


def generate_launch_flag_data(
    launch_data: pd.DataFrame, train_data_end_quarter: str
) -> pd.DataFrame:
    """
    Generate launch flag data for model training.

    Args:
        launch_data (pd.DataFrame): DataFrame containing launch data.
        train_data_end_quarter (str): End quarter of the training data.

    Returns:
        pd.DataFrame: DataFrame containing launch flag data for training.

    Notes:
        - This function preprocesses launch data, aggregates it,
        and generates binary flags for training.
    """

    # Data preprocessing steps for launch_data
    launch_data["J_CODE"] = launch_data["J_CODE"].astype(str)
    launch_data["ASP_MTH"] = pd.to_datetime(launch_data["ASP_MTH"])
    launch_data["QUARTER"] = launch_data["ASP_MTH"].dt.to_period("Q")
    launch_data = launch_data.sort_values(by=["J_CODE", "ASP_MTH", "QUARTER"])
    launch_data = launch_data.reset_index(drop=True)
    launch_data["QUARTER"] = launch_data["QUARTER"].astype(str)

    # Aggregate launch_data based on 'J_CODE' and 'QUARTER'
    launch_data_agg = (
        launch_data.groupby(["J_CODE", "QUARTER"])
        .agg(
            {
                "TIME_SINCE_LAST_COMP_LAUNCH": "first",
                "TIME_TO_NEXT_COMP_LAUNCH": "first",
                "TIME_SINCE_LAST_LOE": "first",
                "TIME_TO_NEXT_LOE": "first",
            }
        )
        .reset_index()
    )

    # Split data into training and testing sets based on quarters
    launch_data_agg_train = launch_data_agg[
        launch_data_agg["QUARTER"] <= train_data_end_quarter
    ]
    launch_data_agg[launch_data_agg["QUARTER"] > train_data_end_quarter]

    # Convert time-related columns to the J_CODE level min value on train data
    launch_data_agg_train = (
        launch_data_agg_train.groupby("J_CODE")
        .agg(
            {
                "TIME_SINCE_LAST_COMP_LAUNCH": "min",
                "TIME_TO_NEXT_COMP_LAUNCH": "min",
                "TIME_SINCE_LAST_LOE": "min",
                "TIME_TO_NEXT_LOE": "min",
            }
        )
        .reset_index()
    )

    # Apply conditions using lambda functions to create binary flags for train data
    launch_data_agg_train["TIME_SINCE_LAST_COMP_LAUNCH"] = launch_data_agg_train[
        "TIME_SINCE_LAST_COMP_LAUNCH"
    ].apply(lambda x: 1 if x <= 3 else 0 if not pd.isna(x) else -1)
    launch_data_agg_train["TIME_TO_NEXT_COMP_LAUNCH"] = launch_data_agg_train[
        "TIME_TO_NEXT_COMP_LAUNCH"
    ].apply(lambda x: 1 if x <= 3 else 0 if not pd.isna(x) else -1)
    launch_data_agg_train["TIME_SINCE_LAST_LOE"] = launch_data_agg_train[
        "TIME_SINCE_LAST_LOE"
    ].apply(lambda x: 1 if x <= 3 else 0 if not pd.isna(x) else -1)
    launch_data_agg_train["TIME_TO_NEXT_LOE"] = launch_data_agg_train[
        "TIME_TO_NEXT_LOE"
    ].apply(lambda x: 1 if x <= 3 else 0 if not pd.isna(x) else -1)

    # Rename columns
    new_column_names = [
        "J_CODE",
        "TIME_SINCE_LAST_COMP_LAUNCH_TRAIN",
        "TIME_TO_NEXT_COMP_LAUNCH_TRAIN",
        "TIME_SINCE_LAST_LOE_TRAIN",
        "TIME_TO_NEXT_LOE_TRAIN",
    ]
    launch_data_agg_train.columns = new_column_names

    return launch_data_agg_train


def transform_data(
    predicted_arima_forecast: pd.DataFrame,
    time_series_columns: list,
    test_months: list,
) -> tuple:
    """
    Transform predicted ARIMA forecast data for model input.

    Args:
        predicted_arima_forecast (pd.DataFrame): Predicted ARIMA forecast data.
        time_series_columns (list): List of time series
        columns to include in transformation.
        test_months (list): List of test months for forecasting.

    Returns:
        Scaler (StandardScaler): Scaler object fitted on the training data.
        transformed_df_scaled (pd.DataFrame): Transformed and scaled forecast data.

    Notes:
        - The function transforms the predicted ARIMA
        forecast data by stacking and unstacking.
        - It scales the data using StandardScaler based on the training months.
    """

    # Subset the relevant columns
    predicted_arima_forecast_subset = predicted_arima_forecast[
        ["J_CODE", "ASP_MTH"] + time_series_columns
    ]

    # Convert ASP_MTH to YR_MTH and set index
    predicted_arima_forecast_subset["YR_MTH"] = predicted_arima_forecast_subset[
        "ASP_MTH"
    ].dt.to_period("M")
    predicted_arima_forecast_subset.set_index(["J_CODE", "YR_MTH"], inplace=True)

    # Stack and unstack the data
    transformed_df = (
        predicted_arima_forecast_subset.stack(dropna=False)
        .unstack(level=1)
        .reset_index()
    )

    # Group by J_CODE and remove first row (NaN after unstacking)
    transformed_df = (
        transformed_df.groupby("J_CODE")
        .apply(lambda x: x.iloc[1:])
        .reset_index(drop=True)
        .rename(columns={"level_1": "TIME_SERIES"})
    )

    # Rename columns to string and set test ASP_MTH values to NaN
    transformed_df.columns = transformed_df.columns.astype(str)
    transformed_df[test_months] = np.nan

    # Sort and reset index
    transformed_df.sort_values(by=["J_CODE", "TIME_SERIES"], inplace=True)
    transformed_df = transformed_df.reset_index(drop=True)
    transformed_df = transformed_df.set_index(["J_CODE", "TIME_SERIES"])
    transformed_df = transformed_df.applymap(lambda x: np.nan if pd.isna(x) else x)

    # Scale the data using StandardScaler
    scaler = StandardScaler()

    transformed_df_scaled = pd.DataFrame(
        scaler.fit_transform(transformed_df.T),
        index=transformed_df.columns,
        columns=transformed_df.index,
    )
    transformed_df_scaled = transformed_df_scaled.T.reset_index()

    return scaler, transformed_df_scaled


def get_market_event_features_for_generic(
    processed_master_data: pd.DataFrame, time_cap: int
) -> pd.DataFrame:
    """
    Extract market event features for generics from processed master data.

    Args:
        processed_master_data (pd.DataFrame): Processed master data
        containing relevant features.
        time_cap (int): Time cap in months.

    Returns:
        pd.DataFrame: DataFrame containing market event features for generics.
    """

    # List of market event features
    market_event_features_list = [
        "TIME_SINCE_LAST_COMP_LAUNCH_IN_MONTHS",
        "TIME_SINCE_LAST_LOE_IN_MONTHS",
        "TIME_SINCE_SAME_CLASS_LAUNCH_IN_MONTHS",
    ]

    # Extract relevant columns from processed master data
    market_event_features_data = processed_master_data[
        ["J_CODE", "ASP_MTH"] + market_event_features_list
    ]

    # Columns to be capped
    capping_cols = [
        "TIME_SINCE_LAST_COMP_LAUNCH_IN_MONTHS",
        "TIME_SINCE_LAST_LOE_IN_MONTHS",
        "TIME_SINCE_SAME_CLASS_LAUNCH_IN_MONTHS",
    ]

    # Apply capping to columns
    for col in capping_cols:
        market_event_features_data[col] = [
            x if x <= time_cap else 0 for x in market_event_features_data[col]
        ]

    return market_event_features_data


def update_event_features_for_non_generic(
    static_feature_data: pd.DataFrame,
    market_event_data: pd.DataFrame,
    train_data_end_quarter: str,
) -> pd.DataFrame:
    """
    Update event features for non-generic products.

    Args:
        static_feature_data (pd.DataFrame): Static feature data for products.
        market_event_data (pd.DataFrame): Market event data.
        train_data_end_quarter (str): End QUARTER of the training data.

    Returns:
        pd.DataFrame: Updated static feature data.

    Notes:
        - This function generates lag-based flag data from market event data
        and merges it with static feature data.
        - It is designed for non-generic products.
    """

    # Generate lag-based flag data from market event data
    lag_based_flag_data = generate_launch_flag_data(
        market_event_data, train_data_end_quarter
    )
    lag_based_flag_data["J_CODE"] = lag_based_flag_data["J_CODE"].astype(str)
    lag_based_flag_data = lag_based_flag_data[
        ["J_CODE", "TIME_SINCE_LAST_COMP_LAUNCH_TRAIN", "TIME_SINCE_LAST_LOE_TRAIN"]
    ]
    lag_based_flag_data = lag_based_flag_data.rename(
        {
            "TIME_SINCE_LAST_COMP_LAUNCH_TRAIN": "TIME_SINCE_LAST_COMP_LAUNCH_TRAIN_FLAG",
            "TIME_SINCE_LAST_LOE_TRAIN": "TIME_SINCE_LAST_LOE_TRAIN_FLAG",
        },
        axis=1,
    )

    # Update data types
    static_feature_data["J_CODE"] = static_feature_data["J_CODE"].astype("object")
    lag_based_flag_data["J_CODE"] = lag_based_flag_data["J_CODE"].astype("object")

    # Merge lag-based flag data with static feature data
    return pd.merge(static_feature_data, lag_based_flag_data, on="J_CODE", how="left")
