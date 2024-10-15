import pandas as pd
import numpy as np
import logging
from utils.helpers import generate_month_year_list
from processing.process_utils import update_ep_data


def process_ep_data(
    ep_data: pd.DataFrame,
    asp_data: pd.DataFrame,
    start_date: str,
    forecast_months: str,
    ep_data_date_cols: list,
    processed_ep_data_cols: list,
    available_codes: list,
    unavailable_codes: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Process EP data to generate additional features
    based on given ASP data and forecast months.

    Args:
        ep_data (pd.DataFrame): DataFrame containing EP data.
        asp_data (pd.DataFrame): DataFrame containing ASP data.
        start_date (str or pd.Timestamp): Start date for forecasting.
        forecast_months (int): Number of months to forecast.
        ep_data_date_cols (list): list of date columns in ep_data.
        processed_ep_data_cols (list): list of columns in processed ep_data.
        available_codes (list): list of j_codes that are available in ep and need to be replaced with corresponding unavailable j_codes.
        unavailable_codes (list): list of j_codes that are unavailable in ep.
        logger (logging.Logger): Logger to capture and log information
        during data processing or model execution.

    Returns:
        pd.DataFrame: Updated EP data with additional features.
    """

    child_logger = logger.getChild("PROCESS_EP_DATA")
    child_logger.info("Processing EP Data...")

    # Read EP Data
    # Adding record for J9330 as a market event happened on 1/1/2007 but this info is not available in EP data.
    new_row = pd.DataFrame({"J_CODE": ["J9330"], "FRST_LAUNCH_USA": ["2007-01-01"]})
    # Add the new row to the DataFrame
    ep_data = pd.concat([ep_data, new_row], ignore_index=True)

    for col in ep_data_date_cols:
        ep_data[col] = pd.to_datetime(ep_data[col])

    # Convert ASP data date column to datetime
    asp_data["CAL_DT"] = pd.to_datetime(asp_data["CAL_DT"])

    # Generate additional months data
    test_months = pd.DataFrame(
        generate_month_year_list(start_date, forecast_months), columns=["CAL_DT"]
    )
    test_months["CAL_DT"] = pd.to_datetime(test_months["CAL_DT"])
    test_months["key"] = 1
    primary_key_cols = ["J_CODE", "PROD_NAME", "CAL_DT"]

    # Merge additional months data with product data
    asp_df = asp_data[primary_key_cols].drop_duplicates()
    df_product = pd.DataFrame({"J_CODE": asp_df["J_CODE"].unique(), "key": 1})
    df_product_test = (
        df_product.merge(test_months, on="key")
        .drop("key", axis=1)
        .sort_values(["J_CODE", "CAL_DT"])
    )
    df_product_test = df_product_test.merge(
        asp_df[["J_CODE", "PROD_NAME"]].drop_duplicates(), on="J_CODE", how="left"
    )

    # Concatenate additional data to the existing data
    asp_df = pd.concat([asp_df, df_product_test], axis=0)
    ep_data.rename(columns={"PROD_NAME": "PRODUCT"}, inplace=True)
    ep_merged = asp_df.merge(ep_data, on="J_CODE", how="left")
    ep_merged["CAL_DT"] = pd.to_datetime(ep_merged["CAL_DT"], utc=True)
    ep_merged["CAL_DT"] = ep_merged["CAL_DT"].dt.tz_localize(None)

    # Feature generation
    date_col_to_consider1 = "FRST_LAUNCH_USA"
    date_col_to_consider2 = "PATENT_EXPRY"

    # Implementation of feature generation
    # Filter records where the date of interest is less than the current date, then calculate time since the last competitive product launch
    ep_filtered_1 = ep_merged[ep_merged[date_col_to_consider1] < ep_merged["CAL_DT"]]
    ep_filtered_1 = (
        ep_filtered_1.groupby(primary_key_cols)[date_col_to_consider1]
        .max()
        .reset_index()
    )
    ep_filtered_1["TIME_SINCE_LAST_COMP_LAUNCH"] = round(
        (ep_filtered_1["CAL_DT"] - ep_filtered_1[date_col_to_consider1]).dt.days / 30
    )

    # Filter records where the date of interest is greater than the current date, then calculate time to the next competitive product launch
    ep_filtered_2 = ep_merged[ep_merged[date_col_to_consider1] > ep_merged["CAL_DT"]]
    ep_filtered_2 = (
        ep_filtered_2.groupby(primary_key_cols)[date_col_to_consider1]
        .min()
        .reset_index()
    )
    ep_filtered_2["TIME_TO_NEXT_COMP_LAUNCH"] = round(
        (ep_filtered_2[date_col_to_consider1] - ep_filtered_2["CAL_DT"]).dt.days / 30
    )

    # Filter records where the date of interest is less than the current date, then calculate time since the last loss of exclusivity (LOE)
    ep_filtered_3 = ep_merged[ep_merged[date_col_to_consider2] < ep_merged["CAL_DT"]]
    ep_filtered_3 = (
        ep_filtered_3.groupby(primary_key_cols)[date_col_to_consider2]
        .max()
        .reset_index()
    )
    ep_filtered_3["TIME_SINCE_LAST_LOE"] = round(
        (ep_filtered_3["CAL_DT"] - ep_filtered_3[date_col_to_consider2]).dt.days / 30
    )

    # Filter records where the LOE date is later than the current date (CAL_DT) and calculate the time to the next LOE.
    ep_filtered_4 = ep_merged[ep_merged[date_col_to_consider2] > ep_merged["CAL_DT"]]
    ep_filtered_4 = (
        ep_filtered_4.groupby(primary_key_cols)[date_col_to_consider2]
        .min()
        .reset_index()
    )
    ep_filtered_4["TIME_TO_NEXT_LOE"] = round(
        (ep_filtered_4[date_col_to_consider2] - ep_filtered_4["CAL_DT"]).dt.days / 30
    )

    # Calculate the time since the last product launch with the same mechanism of action (MoA) within the same class, where the competitive product launch date is earlier than the current date (CAL_DT).
    ep_filtered_temp = ep_merged[
        ep_merged[date_col_to_consider1] <= ep_merged["CAL_DT"]
    ]
    ep_filtered_temp = (
        ep_filtered_temp.groupby(["CAL_DT", "MCHNSM_OF_ACTN"])[date_col_to_consider1]
        .max()
        .reset_index()
    )
    ep_filtered_temp["TIME_SINCE_SAME_CLASS_LAUNCH"] = round(
        (ep_filtered_temp["CAL_DT"] - ep_filtered_temp[date_col_to_consider1]).dt.days
        / 30
    )

    # Merge the mechanism of action (MoA) data with the calculated time since the last same-class launch.
    df3_5 = (
        ep_merged[primary_key_cols + ["MCHNSM_OF_ACTN"]]
        .drop_duplicates()
        .merge(ep_filtered_temp, on=["CAL_DT", "MCHNSM_OF_ACTN"], how="inner")
    )

    # Merge generated features with EP data
    processed_ep_data = ep_merged[primary_key_cols].merge(
        ep_filtered_1, on=primary_key_cols, how="left"
    )
    processed_ep_data = processed_ep_data.merge(
        ep_filtered_2, on=primary_key_cols, how="left"
    )
    processed_ep_data = processed_ep_data.merge(
        ep_filtered_3, on=primary_key_cols, how="left"
    )
    processed_ep_data = processed_ep_data.merge(
        ep_filtered_4, on=primary_key_cols, how="left"
    )
    processed_ep_data = processed_ep_data.merge(df3_5, on=primary_key_cols, how="left")
    processed_ep_data = processed_ep_data[processed_ep_data_cols]
    processed_ep_data = processed_ep_data.drop_duplicates()
    # Call update_ep_data function
    return update_ep_data(processed_ep_data, available_codes, unavailable_codes)


def process_market_event_features(market_event_features: pd.DataFrame) -> pd.DataFrame:
    """
    Process market event features like data obtained from Evaluate Pharma

    Args:
        market_event_features (pd.DataFrame): Market event features dataframe.

    Returns:
        pd.DataFrame: Processed market event features dataframe.

    Note:
        - The market_event_features dataframe should contain the following columns:
            - CAL_DT: Date of the market event.
            - J_CODE: Code corresponding to the market event.
            - TIME_SINCE_LAST_COMP_LAUNCH: Time since the last competitor launch.
            - TIME_TO_NEXT_COMP_LAUNCH: Time to the next competitor launch.
            - TIME_SINCE_LAST_LOE: Time since the last loss of exclusivity.
            - TIME_TO_NEXT_LOE: Time to the next loss of exclusivity.
            - TIME_SINCE_SAME_CLASS_LAUNCH: Time since the last launch in the same cls.

    """

    # Process and sort market event feature data

    # Convert the "CAL_DT" column to datetime and rename it to "ASP_MTH."
    market_event_features_processed = market_event_features.copy()

    market_event_features_processed["CAL_DT"] = pd.to_datetime(
        market_event_features_processed["CAL_DT"]
    )
    market_event_features_processed.rename(columns={"CAL_DT": "ASP_MTH"}, inplace=True)

    # Ensure "J_CODE" is treated as a string for consistency.
    market_event_features_processed["J_CODE"] = market_event_features_processed[
        "J_CODE"
    ].astype(str)

    # Sort the data by "J_CODE" and "ASP_MTH" and reset the index.
    market_event_features_processed = market_event_features_processed.sort_values(
        by=["J_CODE", "ASP_MTH"]
    )
    market_event_features_processed = market_event_features_processed.reset_index(
        drop=True
    )

    # Group by "J_CODE" and "ASP_MTH," selecting the first occurrence of key event features (time since/to competitor launches and LOEs).
    market_event_features_processed = (
        market_event_features_processed.groupby(["J_CODE", "ASP_MTH"])
        .agg(
            {
                "TIME_SINCE_LAST_COMP_LAUNCH": "first",
                "TIME_TO_NEXT_COMP_LAUNCH": "first",
                "TIME_SINCE_LAST_LOE": "first",
                "TIME_TO_NEXT_LOE": "first",
                "TIME_SINCE_SAME_CLASS_LAUNCH": "first",
            }
        )
        .reset_index()
    )

    # Sort the processed market event features by "J_CODE" and "ASP_MTH" and reset the index.
    market_event_features_processed = market_event_features_processed.sort_values(
        by=["J_CODE", "ASP_MTH"]
    )
    return market_event_features_processed.reset_index(drop=True)


def process_master_data(
    master_data: pd.DataFrame,
    data_market_event_features: pd.DataFrame,
    master_data_ordered_cols: list,
    master_data_date_cols: list,
    market_event_cols: list,
) -> pd.DataFrame:
    """
    Process master data and merge with market event features.

    Args:
        master_data (pd.DataFrame): DataFrame containing master data.
        data_market_event_features (pd.DataFrame): df containing market event features.
        master_data_ordered_cols (list): list of columns of master data in req. order.
        master_data_date_cols (list): list of date columns of master data.
        market_event_cols (list): list of market event features

    Returns:
        pd.DataFrame: Processed master data DataFrame.

    Notes:
        - The function performs the following processing steps:
            1. Orders columns in the master data DataFrame.
            2. Renames the 'CAL_DT' column to 'ASP_MTH'.
            3. Converts date columns to datetime format.
            4. Groups the DataFrame by 'J_CODE' and 'ASP_MTH' and selects the first row.
            5. Calculates quarterly ASP (Average Selling Price) from ASP_PRC column.
            6. Calculates time-based features such as time since launch date,
            time difference from patent expiry, and time since generic launch date.
            7. Merges the processed master data with market event features.
    """

    # Order columns in master data DataFrame
    master_df_filtered = master_data[master_data_ordered_cols]

    # drop column ASP_MTH
    master_df_filtered.drop(columns=["ASP_MTH"], inplace=True)

    # Convert date columns to datetime format
    master_df_filtered[master_data_date_cols] = master_df_filtered[
        master_data_date_cols
    ].apply(pd.to_datetime, errors="coerce")

    master_df_filtered.rename(columns={"CAL_DT": "ASP_MTH"}, inplace=True)
    master_df_filtered["J_CODE"] = master_df_filtered["J_CODE"].astype(str)
    # Group DataFrame by 'J_CODE' and 'ASP_MTH' and select first row
    master_df_filtered = (
        master_df_filtered.groupby(["J_CODE", "ASP_MTH"]).head(1).reset_index(drop=True)
    )
    master_df_filtered = master_df_filtered.sort_values(by=["J_CODE", "ASP_MTH"])
    master_df_filtered = master_df_filtered.reset_index(drop=True)

    # Calculate quarterly ASP from ASP_PRC column

    # Convert "ASP_MTH" to a quarterly period.
    master_df_filtered["QUARTER"] = master_df_filtered["ASP_MTH"].dt.to_period("Q")

    # Forward-fill missing values in "ASP_PRC" within each "J_CODE" and "QUARTER" group.
    master_df_filtered["ASP_PRC"] = master_df_filtered.groupby(["J_CODE", "QUARTER"])[
        "ASP_PRC"
    ].transform(lambda x: x.fillna(method="ffill"))

    # Backward-fill missing values in "ASP_PRC" within each "J_CODE" and "QUARTER" group.
    master_df_filtered["ASP_PRC"] = master_df_filtered.groupby(["J_CODE", "QUARTER"])[
        "ASP_PRC"
    ].transform(lambda x: x.fillna(method="bfill"))

    # Extract the month from "ASP_MTH" and compute the difference from the quarter.
    master_df_filtered["ASP_MTH_Q"] = (
        master_df_filtered["ASP_MTH"].map(lambda x: str(x).split("-")[1]).astype(int)
    )
    master_df_filtered["ASP_MTH_Q"] = master_df_filtered[
        "ASP_MTH_Q"
    ] - master_df_filtered["QUARTER"].map(
        master_df_filtered.groupby(["QUARTER"])["ASP_MTH_Q"].min()
    )

    # Set "ASP_TRUE" to "ASP_PRC" if "ASP_MTH_Q" is 0, otherwise NaN.
    master_df_filtered["ASP_TRUE"] = master_df_filtered.apply(
        lambda x: x["ASP_PRC"] if x["ASP_MTH_Q"] == 0 else np.nan, axis=1
    )

    # Drop the "ASP_PRC" column as it's no longer needed.
    master_df_filtered.drop(columns=["ASP_PRC"], inplace=True)

    # Replace "null" strings in "ASP_TRUE" with NaN.
    master_df_filtered["ASP_TRUE"] = master_df_filtered["ASP_TRUE"].replace(
        "null", np.nan
    )

    # Interpolate missing values in "ASP_TRUE" linearly within each "J_CODE".
    master_df_filtered["ASP_TRUE"] = (
        master_df_filtered.groupby(["J_CODE"])
        .apply(lambda x: x["ASP_TRUE"].astype(float).interpolate(method="linear"))
        .reset_index()["ASP_TRUE"]
        .values
    )

    # Calculate time-based features

    # Group the dataframe by 'J_CODE'
    grouped = master_df_filtered.groupby("J_CODE")

    # Calculate time since the latest launch date within each 'J_CODE' group
    master_df_filtered["TIME_SINCE_LAUNCH_DATE"] = grouped.apply(
        lambda x: (x["ASP_MTH"] - x["TNTV_LAUNCH_DT"].max()).dt.days
    ).reset_index(drop=True)

    # Calculate time difference between the latest patent expiry and ASP month within each 'J_CODE' group
    master_df_filtered["TIME_DIFF_PATENT_EXPIRY"] = grouped.apply(
        lambda x: (x["PATENT_EXPRY"].max() - x["ASP_MTH"]).dt.days
    ).reset_index(drop=True)

    # Calculate time since the latest generic launch date within each 'J_CODE' group
    master_df_filtered["TIME_SINCE_GENERIC_LAUNCH_DATE"] = grouped.apply(
        lambda x: (x["ASP_MTH"] - x["TNTV_GNRC_LAUNCH_DT"].max()).dt.days
    ).reset_index(drop=True)
    del grouped

    # Convert 'J_CODE' in both 'master_df_filtered' and 'data_market_event_features' to string and then object types to ensure consistency and treat them as categorical variables for efficient comparisons and potential future operations.
    master_df_filtered["J_CODE"] = (
        master_df_filtered["J_CODE"].astype("str").astype("object")
    )
    data_market_event_features["J_CODE"] = (
        data_market_event_features["J_CODE"].astype("str").astype("object")
    )
    # Forward-fill missing values in 'ASP_TRUE' column
    master_df_filtered["ASP_TRUE"] = master_df_filtered.groupby("J_CODE")[
        "ASP_TRUE"
    ].transform(lambda x: x.fillna(method="ffill"))

    # Calculate ASP growth
    # Create a lagged version of 'ASP_TRUE' to capture the previous month's ASP for each 'J_CODE', calculate the ASP growth by dividing the current ASP by the lagged ASP, and then drop the temporary 'ASP_LAGGED' column to clean up the dataframe.
    master_df_filtered["ASP_LAGGED"] = (
        master_df_filtered.sort_values(["J_CODE", "ASP_MTH"])
        .groupby(["J_CODE"])["ASP_TRUE"]
        .shift(1)
    )
    master_df_filtered["ASP_GROWTH"] = (
        master_df_filtered["ASP_TRUE"] / master_df_filtered["ASP_LAGGED"]
    )
    master_df_filtered.drop(columns=["ASP_LAGGED"], inplace=True)

    # Merge processed master data with market event features
    return pd.merge(
        master_df_filtered,
        data_market_event_features[market_event_cols].rename(
            {
                "TIME_SINCE_LAST_COMP_LAUNCH": "TIME_SINCE_LAST_COMP_LAUNCH_IN_MONTHS",
                "TIME_SINCE_LAST_LOE": "TIME_SINCE_LAST_LOE_IN_MONTHS",
                "TIME_SINCE_SAME_CLASS_LAUNCH": "TIME_SINCE_SAME_CLASS_LAUNCH_IN_MONTHS",
            },
            axis=1,
        ),
        on=["J_CODE", "ASP_MTH"],
        how="outer",
    )


def create_train_test_month_quarters(
    start_forecast_month: str,
    forecast_months: int,
    processed_master_data: pd.DataFrame,
    logger: logging.Logger,
) -> tuple:
    """
    Create train and test months and quarters based on the
    forecast start month and duration.

    Args:
        start_forecast_month (str): Start month for the forecast in 'YYYY-MM' format.
        forecast_months (int): Number of forecast months.
        processed_master_data (pd.DataFrame): Processed master data containing the 'ASP_MTH' column.
        logger (logging.Logger): Logger to capture and log information during data processing or model execution.

    Returns:
        train_months (List[str]): List of train months in 'YYYY-MM' format.
        test_months (List[str]): List of test months in 'YYYY-MM' format.
        train_quarters (List[str]): List of train quarters in 'YYYY-Q' format.
        test_quarters (List[str]): List of test quarters in 'YYYY-Q' format.

    Raises:
        ValueError: If the start_forecast_month format is invalid
        or forecast_months is non-positive.

    Notes:
        - The train months are calculated based on the available
        data up to the forecast start month.
        - The test months are generated starting from the forecast start month.
    """

    child_logger = logger.getChild("TRAIN_TEST_MONTH_DATA")
    child_logger.info(
        " Master dataframe Information:{} ".format(processed_master_data.info())
    )
    # Validate forecast months
    if forecast_months <= 0:
        raise ValueError("forecast_months must be a positive integer.")

    end_date = pd.to_datetime(start_forecast_month) + pd.DateOffset(
        months=forecast_months - 1
    )

    all_months = processed_master_data[processed_master_data["ASP_MTH"] < end_date][
        "ASP_MTH"
    ].unique()

    # Convert the list of all months to numpy array and then to datetime64 format with month precision.
    all_months = np.array(all_months)
    all_months = all_months.astype("datetime64[M]")

    # Convert datetime64 months to string format in 'YYYY-MM' and truncate to the first 7 characters.
    all_months = [np.datetime_as_string(date, unit="M")[:7] for date in all_months]

    # Generate a list of forecast months and determine the months used for training by excluding test months.
    test_months = generate_month_year_list(start_forecast_month, forecast_months)
    train_months = sorted(list(set(set(all_months) - set(test_months))))

    # Convert train months to quarterly periods and sort them.
    train_quarters = pd.to_datetime(train_months, format="%Y-%m").to_period("Q")
    train_quarters = sorted(list(set(train_quarters.astype("str"))))

    # Convert test months to quarterly periods and sort them.
    test_quarters = pd.to_datetime(test_months, format="%Y-%m").to_period("Q")
    test_quarters = sorted(list(set(test_quarters.astype("str"))))

    # Log the train and test quarters for verification.
    child_logger.info("===========================================")
    child_logger.info("Train quarters:{}".format(train_quarters))
    child_logger.info("===========================================")
    child_logger.info("Test quarters:{}".format(test_quarters))
    child_logger.info("===========================================")

    child_logger.info(
        "Forecast quarters:{}".format(
            sorted(list(set(test_quarters) - set(train_quarters)))
        )
    )
    child_logger.info("===========================================")

    return train_months, test_months, train_quarters, test_quarters
