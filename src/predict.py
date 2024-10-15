import pandas as pd
import logging


def get_xgb_predictions(
    window_wise_feature_data: pd.DataFrame,
    model,
    window_size: int,
    processed_master_data: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    test_months: list,
    forecast_months: int,
    time_series_columns: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Get XGBoost predictions for the specified forecast period.

    Args:
        window_wise_feature_data (pd.DataFrame): Window-wise feature data.
        model: Trained XGBoost model.
        window_size (int): Size of the sliding window.
        processed_master_data (pd.DataFrame): Processed master data.
        X_train (pd.DataFrame): Features of the training data.
        y_train (pd.DataFrame): Target variables of the training data.
        test_months (list): List of months to be used for testing.
        forecast_months (int): Number of months to forecast.
        time_series_columns (list): List of time series columns.

    Returns:
        pd.DataFrame: DataFrame containing XGBoost predictions.

    Notes:
        - This function generates predictions using the trained
        XGBoost model for the specified forecast period.
        - It iterates over the forecast period, updating the predictions
        and window-wise feature data accordingly.
    """

    exclude_columns = [
        "J_CODE",
        "WINDOW_START",
        "WINDOW_END",
        "TARGET_MONTH",
    ] + time_series_columns

    predictions = []

    window_wise_feature_data_test_i = window_wise_feature_data[
        window_wise_feature_data.index.get_level_values("TARGET_MONTH")
        == test_months[0]
    ]

    for i in range(forecast_months):
        # Drop irrelevant columns
        window_wise_feature_data_test_i = window_wise_feature_data_test_i.drop(
            columns=[
                c
                for c in window_wise_feature_data_test_i.columns
                if c in exclude_columns
            ]
        )
        # Make predictions
        y_pred = model.predict(window_wise_feature_data_test_i)
        y_pred = pd.DataFrame(
            data=y_pred,
            index=window_wise_feature_data_test_i.index,
            columns=y_train.columns,
        )
        predictions.append(y_pred)

        # Update data for the next prediction
        y_pred_tt = y_pred.reset_index().copy()
        y_pred_tt["TARGET_MONTH"] = pd.to_datetime(y_pred_tt["TARGET_MONTH"])
        y_pred_tt["UPDATE_MONTH"] = y_pred_tt["TARGET_MONTH"] + pd.DateOffset(months=1)
        y_pred_tt["UPDATE_MONTH"] = (
            y_pred_tt["UPDATE_MONTH"].dt.to_period("M").astype(str)
        )
        y_pred_tt = y_pred_tt.set_index(["J_CODE", "UPDATE_MONTH"]).drop(
            columns=["TARGET_MONTH"]
        )

        window_wise_feature_data_test_prev_month = (
            window_wise_feature_data_test_i.copy()
        )
        cols_to_update = [
            c
            for c in window_wise_feature_data_test_i.columns
            if (str(window_size - 1) + "_") in c and ("USC5_DSCR" not in c)
        ]

        try:
            window_wise_feature_data_test_i = window_wise_feature_data[
                window_wise_feature_data.index.get_level_values("TARGET_MONTH")
                == test_months[i + 1]
            ]
        except IndexError:
            break

        y_pred_tt = y_pred_tt.loc[window_wise_feature_data_test_i.index]
        window_wise_feature_data_test_i[cols_to_update] = y_pred_tt[["RESIDUAL"]].values

        col_idx = window_wise_feature_data_test_i.isna().sum()
        col_idx = col_idx[col_idx == processed_master_data.J_CODE.nunique()].index
        col_idx = [c for c in col_idx if c in X_train.columns]

        if len(col_idx) > 0:
            past_col_idx = [
                str(int(c.split("_")[0]) + 1) + "_" + "_".join(c.split("_")[1:])
                for c in col_idx
            ]
            window_wise_feature_data_test_i[
                col_idx
            ] = window_wise_feature_data_test_prev_month.loc[
                window_wise_feature_data_test_i.index.get_level_values("J_CODE")
            ][
                past_col_idx
            ].values

        common_columns = [
            col
            for col in X_train.columns
            if col in window_wise_feature_data_test_i.columns
        ]
        col_idx = window_wise_feature_data_test_i[common_columns].isna().sum()
        col_idx = col_idx[col_idx > 0]

    xgb_predicted_data = pd.concat(predictions)
    return xgb_predicted_data.reset_index()


def get_xgb_predictions_inference(
    window_wise_feature_data: pd.DataFrame,
    model,
    window_size: int,
    processed_master_data: pd.DataFrame,
    forecast_months: int,
    time_series_columns: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Get XGBoost predictions for the specified forecast period using the trained model.

    Args:
        window_wise_feature_data (pd.DataFrame): Window-wise feature data.
        model: Trained XGBoost model.
        window_size (int): Size of the sliding window.
        processed_master_data (pd.DataFrame): Processed master data.
        forecast_months (int): Number of months to forecast.
        time_series_columns (list): List of time series columns.

    Returns:
        pd.DataFrame: DataFrame containing XGBoost predictions.

    Notes:
        - This function generates predictions using the trained
        XGBoost model for the specified forecast period.
        - It iterates over the forecast period, updating the
        predictions and window-wise feature data accordingly.
    """

    exclude_columns = [
        "J_CODE",
        "WINDOW_START",
        "WINDOW_END",
        "TARGET_MONTH",
    ] + time_series_columns

    predictions = []
    current_data = window_wise_feature_data
    initial_data = window_wise_feature_data[
        window_wise_feature_data.index.get_level_values("TARGET_MONTH")
        == window_wise_feature_data.index.get_level_values("TARGET_MONTH").min()
    ]
    for i in range(forecast_months):
        # Drop irrelevant columns
        current_data = current_data.drop(
            columns=[c for c in current_data.columns if c in exclude_columns]
        )
        # Make predictions
        y_pred = model.predict(current_data)
        y_pred = pd.DataFrame(
            data=y_pred,
            index=current_data.index,
            columns=["RESIDUAL"],
        )
        predictions.append(y_pred)

        # Update data for the next prediction
        y_pred_tt = y_pred.reset_index().copy()
        y_pred_tt["UPDATE_MONTH"] = (
            (
                pd.to_datetime(y_pred_tt.index.get_level_values("TARGET_MONTH"))
                + pd.DateOffset(months=1)
            )
            .dt.to_period("M")
            .astype(str)
        )
        y_pred_tt = y_pred_tt.set_index(["J_CODE", "UPDATE_MONTH"]).drop(
            columns=["TARGET_MONTH"]
        )

        # Prepare the next month's data
        if i + 1 < forecast_months:
            try:
                next_month_data = window_wise_feature_data[
                    window_wise_feature_data.index.get_level_values("TARGET_MONTH")
                    == (
                        pd.to_datetime(y_pred_tt["UPDATE_MONTH"])
                        .dt.to_period("M")
                        .astype(str)
                    ).iloc[0]
                ]

                # Ensure the new data includes the window size
                if len(next_month_data) >= window_size:
                    next_month_data = next_month_data.tail(window_size)
                else:
                    next_month_data = pd.concat([initial_data, next_month_data]).tail(
                        window_size
                    )

                next_month_data["RESIDUAL"] = y_pred_tt["RESIDUAL"].values

                # Align with processed_master_data structure if needed
                if "J_CODE" in processed_master_data.columns:
                    next_month_data = next_month_data.reindex(
                        processed_master_data.index, fill_value=0
                    )
                current_data = next_month_data
            except KeyError:
                break

    return pd.concat(predictions).reset_index()


def aggregate_predictions(
    xgb_pred_data: pd.DataFrame,
    arima_predicted_data: pd.DataFrame,
    scaler: object,
) -> pd.DataFrame:
    """
    Function to aggregate XGBoost and ARIMA predicted data and perform post-processing.

    Args:
        xgb_pred_data (pd.DataFrame): XGBoost predicted data with
        columns "TARGET_MONTH", "J_CODE", "RESIDUAL".
        arima_predicted_data (pd.DataFrame): ARIMA predicted data with
        columns "J_CODE", "ASP_MTH", "ARIMA_PREDICTED_ASP".
        scaler (object): Scaler object used for scaling the data.

    Returns:
        pd.DataFrame: Aggregated and post-processed predicted data with
        columns "J_CODE", "ASP_MTH", "RESIDUAL_PREDICTED_XGB",
        "ARIMA_PREDICTED_ASP", "FINAL_PREDICTED_ASP".
    """

    # Rename columns
    xgb_pred_data = xgb_pred_data.rename(
        {"TARGET_MONTH": "ASP_MTH", "RESIDUAL": "RESIDUAL_PREDICTED_XGB"}, axis=1
    )

    # Convert "ASP_MTH" column to datetime
    xgb_pred_data["ASP_MTH"] = pd.to_datetime(xgb_pred_data["ASP_MTH"])

    # Sort by "J_CODE" and "ASP_MTH"
    xgb_pred_data.sort_values(["J_CODE", "ASP_MTH"], inplace=True)

    # Add "YR_MTH" column
    xgb_pred_data["YR_MTH"] = xgb_pred_data["ASP_MTH"].dt.to_period("M")

    # Set index as "J_CODE" and "YR_MTH"
    xgb_pred_data.set_index(["J_CODE", "YR_MTH"], inplace=True)

    # Perform data transformation
    xgb_pred_data_transformed = (
        xgb_pred_data.stack(dropna=False).unstack(level=1).reset_index()
    )

    xgb_pred_data_transformed = (
        xgb_pred_data_transformed.groupby("J_CODE")
        .apply(lambda x: x.iloc[1:])
        .reset_index(drop=True)
        .rename(columns={"level_1": "TIME_SERIES"})
    )

    xgb_pred_data_transformed.columns = xgb_pred_data_transformed.columns.astype(str)
    xgb_pred_data_transformed.sort_values(by=["J_CODE", "TIME_SERIES"], inplace=True)
    xgb_pred_data_transformed = xgb_pred_data_transformed.reset_index(drop=True)
    xgb_pred_data_transformed = xgb_pred_data_transformed.set_index(
        ["J_CODE", "TIME_SERIES"]
    )

    # Perform data inverse scaling
    xgb_pred_data_inverse_scaled = pd.DataFrame(
        scaler.inverse_transform(xgb_pred_data_transformed.T),
        index=xgb_pred_data_transformed.columns,
        columns=xgb_pred_data_transformed.index,
    )
    xgb_pred_data_inverse_scaled = xgb_pred_data_inverse_scaled.T

    # Reshape the data
    xgb_pred_data_inverse_scaled_retransformed = xgb_pred_data_inverse_scaled.stack(
        dropna=False
    ).reset_index()
    xgb_pred_data_inverse_scaled_retransformed.columns = [
        "J_CODE",
        "TIME_SERIES",
        "ASP_MTH",
        "RESIDUAL_PREDICTED_XGB",
    ]
    xgb_pred_data_inverse_scaled_retransformed["ASP_MTH"] = pd.to_datetime(
        xgb_pred_data_inverse_scaled_retransformed["ASP_MTH"]
    )
    xgb_pred_data_inverse_scaled_retransformed.drop(
        columns=["TIME_SERIES"], inplace=True
    )
    xgb_pred_data_inverse_scaled_retransformed = (
        xgb_pred_data_inverse_scaled_retransformed.set_index(["J_CODE", "ASP_MTH"])
    )
    xgb_pred_data_inverse_scaled_retransformed = (
        xgb_pred_data_inverse_scaled_retransformed.reset_index()
    )
    xgb_pred_data_inverse_scaled_retransformed = (
        xgb_pred_data_inverse_scaled_retransformed.set_index(["J_CODE", "ASP_MTH"])
    )

    # Set index of ARIMA predicted data
    arima_predicted_data = arima_predicted_data.set_index(["J_CODE", "ASP_MTH"])

    # Merge XGBoost and ARIMA predicted data
    prediction_post_processed = pd.merge(
        xgb_pred_data_inverse_scaled_retransformed.reset_index(),
        arima_predicted_data.reset_index(),
        on=["J_CODE", "ASP_MTH"],
        how="left",
    )

    # Drop unnecessary columns
    prediction_post_processed.drop(columns=["RESIDUAL"], inplace=True)

    # Calculate final predicted ASP
    prediction_post_processed["FINAL_PREDICTED_ASP"] = (
        prediction_post_processed["ARIMA_PREDICTED_ASP"]
        + prediction_post_processed["RESIDUAL_PREDICTED_XGB"]
    )

    return prediction_post_processed


def retransform_predictions(
    aggregated_predictions: pd.DataFrame,
    processed_master_data: pd.DataFrame,
    generic_flag: bool,
) -> pd.DataFrame:
    """
    Retransform aggregated predictions to obtain final forecasted ASP.

    Args:
        aggregated_predictions (pd.DataFrame): df containing aggregated predictions.
        processed_master_data (pd.DataFrame): Processed master data.
        generic_flag (bool): A flag indicating whether generic features are used.

    Returns:
        pd.DataFrame: DataFrame containing retransformed predictions.

    Notes:
        - This function retransforms aggregated predictions
        to obtain final forecasted ASP.
        - If generic_flag is True, it adjusts the final predicted
        ASP based on the previous true ASP values.
    """

    if generic_flag:
        if "FINAL_PREDICTED_ASP" not in aggregated_predictions.columns:
            aggregated_predictions["FINAL_PREDICTED_ASP"] = None
        group_list = []
        for name, group in aggregated_predictions.groupby("J_CODE"):
            first_row_index = group.index[0]

            # Filter the processed_master_data DataFrame
            filtered_data = processed_master_data[
                (processed_master_data["J_CODE"] == name)
                & (
                    processed_master_data["ASP_MTH"]
                    == group.ASP_MTH.iloc[0] + pd.DateOffset(months=-1)
                )
            ]
            # Check if the filtered DataFrame is not empty
            # and contains the necessary column
            if (
                not filtered_data.empty
                and "ASP_TRUE" in filtered_data.columns
                and len(filtered_data["ASP_TRUE"].values) > 0
            ):
                asp_true_value = filtered_data["ASP_TRUE"].values[0]
            else:
                # Handle the case where filtered data is empty or column is missing
                asp_true_value = 1  # or set to a default value or raise an exception
                # Optionally raise an exception or log an error message
                print(
                    f"""No matching data found for J_CODE={name} and
                    Month={group.ASP_MTH.iloc[0] + pd.DateOffset(months=-1)}
                    . Skipping update."""
                )

            # Update the group DataFrame
            group.at[first_row_index, "FINAL_PREDICTED_ASP"] = (
                group.at[first_row_index, "FINAL_PREDICTED_ASP"] * asp_true_value
            )

            for i in group.index[1:]:
                group.loc[i, "FINAL_PREDICTED_ASP"] = (
                    group.loc[i - 1, "FINAL_PREDICTED_ASP"]
                    * group.loc[i, "FINAL_PREDICTED_ASP"]
                )
            group_list.append(group)

        predictions_retransformed = pd.concat(group_list)
        predictions_retransformed = predictions_retransformed.rename(
            {"ASP_TRUE": "ACTUAL_ASP_NEW"}, axis=1
        )
        predictions_retransformed = (
            predictions_retransformed.groupby(
                ["J_CODE", predictions_retransformed["ASP_MTH"].dt.to_period("Q")]
            )
            .agg({"FINAL_PREDICTED_ASP": "mean", "ACTUAL_ASP_NEW": "first"})
            .reset_index()
        )
    else:
        predictions_retransformed = (
            aggregated_predictions.groupby(
                ["J_CODE", aggregated_predictions["ASP_MTH"].dt.to_period("Q")]
            )
            .agg({"FINAL_PREDICTED_ASP": "mean", "ACTUAL_ASP": "first"})
            .reset_index()
        )

    predictions_retransformed.columns = [
        "J_CODE",
        "TIME",
        "ASP_FORECASTED_MEAN",
        "ACTUAL",
    ]
    predictions_retransformed.sort_values(["J_CODE", "TIME"], inplace=True)

    return predictions_retransformed
