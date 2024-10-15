import pandas as pd
import numpy as np


def update_ep_data(
    ep_data_processed: pd.DataFrame,
    available_in_ep: list,
    not_available_in_ep: list,
) -> pd.DataFrame:
    """
    Update the ep_data_processed DataFrame by replacing
    certain values in the 'J_CODE' column.

    Args:
        ep_data_processed (pd.DataFrame): DataFrame containing processed ep_data.
        available_in_ep (list): List of j_codes that are available in ep and need to be replaced with corresponding unavailable j_codes.
        not_available_in_ep (list): List of j_codes that are unavailable in ep.

    Returns:
        pd.DataFrame: Updated ep_data_processed DataFrame.

    Notes:
        - The function replaces values in the 'J_CODE' column as follows:
        - Replaces values in the 'J_CODE' column with values from 'available_in_ep'
        list with corresponding values from the 'not_available_in_ep' list.
    """

    ep_data_replace = ep_data_processed.copy()

    # Exclude 'not_available_in_ep' values from DataFrame
    ep_data_replace = ep_data_replace[
        ~ep_data_replace["J_CODE"].isin(not_available_in_ep)
    ]

    # Replace 'available_in_ep' values with corresponding 'not_available_in_ep' values
    ep_data_replace_replacement = ep_data_replace[
        ep_data_replace["J_CODE"].isin(available_in_ep)
    ]
    ep_data_replace_replacement["J_CODE"] = ep_data_replace_replacement[
        "J_CODE"
    ].replace(available_in_ep, not_available_in_ep)

    # Concatenate updated DataFrame with replaced values
    return pd.concat([ep_data_replace, ep_data_replace_replacement], axis=0)


def identify_jcodes_constant_forecast(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    train_qrtr: list,
    forecast_qrtr: list,
) -> list:
    """
    Identify J_CODES with constant forecast for 8 quarters and
    first quarter forecast same as last quarter actual.

    Args:
        df_actual_train_val (pandas.DataFrame): DataFrame containing actual training/validation data.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        train_qrtr (list): List of training/validation quarters.
        forecast_qrtr (list): List of forecast quarters.

    Returns:
        list: List of J_CODES with constant 8-quarter forecast and
        first quarter forecast same as last quarter actual.
    """

    # Identifying the J_CODES with last quarter actual same as first quarter forecast

    # Extracting last quarter actual
    df_actual_lastQ = df_actual_train_val.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_TRUE"
    )[[train_qrtr[-1]]]

    # Extracting first quarter forecast
    df_model_forecast_firstQ = df_model_forecast.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_PRED"
    )[[forecast_qrtr[0]]]

    # Merging dataframes with last quarter actual and first quarter forecast
    df_actual_lastQ_forecast_firstQ = df_actual_lastQ.join(df_model_forecast_firstQ)

    # Drop the dataframes
    del df_actual_lastQ, df_model_forecast_firstQ

    # Identify J_CODES with constant 8-quarter forecast
    J_CODE_constant_forecast = (
        df_model_forecast.groupby("J_CODE")["ASP_PRED"]
        .nunique()[
            (df_model_forecast.groupby("J_CODE")["ASP_PRED"].nunique() == 1).values
        ]
        .index
    )

    # Identify J_CODES with first quarter forecast same as last quarter actual
    df_actual_lastQ_forecast_firstQ_SAME = df_actual_lastQ_forecast_firstQ[
        df_actual_lastQ_forecast_firstQ[train_qrtr[-1]]
        == df_actual_lastQ_forecast_firstQ[forecast_qrtr[0]]
    ]

    # Dropping the J_CODES with constant forecast
    df_last_actual_copied = df_actual_lastQ_forecast_firstQ_SAME[
        df_actual_lastQ_forecast_firstQ_SAME.index.isin(J_CODE_constant_forecast)
    ]
    del J_CODE_constant_forecast, df_actual_lastQ_forecast_firstQ_SAME

    # Extract the J_CODES
    return list(df_last_actual_copied.index)


def identify_jcodes_with_highdrop(
    df_model_forecast: pd.DataFrame, forecast_qrtr: list
) -> list:
    """
    Identify J_CODES with a significant drop (more than 20%) in forecast.

    Args:
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        forecast_qrtr (list): List of forecast quarters.

    Returns:
        list: List of J_CODES with a significant drop in forecast.
    """

    # Pivot the forecast dataframe
    df_model_forecast_pivot = df_model_forecast.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_PRED"
    )

    # Calculate the growth of forecast over the 8 quarters which is calculated as ([last_quarter_forecast]/[first_quarter_forecast])/8
    df_model_forecast_pivot["GROWTH_OF_FORECAST"] = (
        df_model_forecast_pivot[forecast_qrtr[-1]]
        / df_model_forecast_pivot[forecast_qrtr[0]]
    ) ** (1 / 8)

    # Identify J_CODES with more than 20% drop in forecast
    return list(
        df_model_forecast_pivot[
            df_model_forecast_pivot["GROWTH_OF_FORECAST"] < 0.8
        ].index
    )


def define_start_end_point(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    j_lastQ_actual_copied: list,
    j_high_drop: list,
    train_qrtr: list,
    forecast_qrtr: list,
) -> pd.DataFrame:
    """
    Define start and end points for forecasting based on
    copied actuals and high drop forecasts.

    Args:
        df_actual_train_val (pandas.DataFrame): DataFrame containing actual training/validation data.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        j_lastQ_actual_copied (list): List of J_CODES with last quarter actuals copied.
        j_high_drop (list): List of J_CODES with high drop forecasts.
        train_qrtr (list): List of training/validation quarters.
        forecast_qrtr (list): List of forecast quarters.

    Returns:
        pandas.DataFrame: DataFrame containing start and end points for forecasting.
    """

    # Calculate growth of last quarter actuals
    df_lastQ_growth_actual = df_actual_train_val.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_TRUE"
    ).loc[j_lastQ_actual_copied, [train_qrtr[-2], train_qrtr[-1]]]
    df_lastQ_growth_actual = (
        df_lastQ_growth_actual[train_qrtr[-1]] / df_lastQ_growth_actual[train_qrtr[-2]]
    )

    # Calculate end point for J_CODES with last quarter actuals copied

    # Pivot the dataframe to get the ASP_TRUE values with 'J_CODE' as index and 'QUARTER' as columns
    df_end_point_lastQ_actual_copied = (
        df_actual_train_val.pivot(index="J_CODE", columns="QUARTER", values="ASP_TRUE")
        .loc[j_lastQ_actual_copied, [train_qrtr[-1]]]
        .join(df_lastQ_growth_actual.to_frame())
    )

    # Compute the adjusted values by multiplying the last quarter's ASP_TRUE with the growth actual values
    df_end_point_lastQ_actual_copied = (
        df_end_point_lastQ_actual_copied[train_qrtr[-1]]
        * df_end_point_lastQ_actual_copied[0]
    )

    # Extract end point for J_CODES with high drop forecasts
    df_end_point_high_drop = df_model_forecast.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_PRED"
    ).loc[j_high_drop][forecast_qrtr[-1]]

    # Concatenate end points
    df_end_all = pd.concat([df_end_point_lastQ_actual_copied, df_end_point_high_drop])

    # Extract start points for all J_CODES
    df_start_all = df_actual_train_val.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_TRUE"
    ).loc[j_lastQ_actual_copied + j_high_drop][train_qrtr[-1]]

    # Concatenate start and end points
    df_start_end_all = df_start_all.to_frame().join(df_end_all.to_frame())
    df_start_end_all.columns = [train_qrtr[-1], forecast_qrtr[-1]]

    return df_start_end_all


def impute_constant_drop(
    df_start_end_all: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    j_lastQ_actual_copied: list,
    j_high_drop: list,
    train_qrtr: list,
    forecast_qrtr: list,
) -> pd.DataFrame:
    """
    Impute constant drop in forecasts using geometric and linear growth.

    Args:
        df_start_end_all (pd.DataFrame): df containing start and end points for forecasting.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        j_lastQ_actual_copied (list): List of J_CODES with last quarter actuals copied.
        j_high_drop (list): List of J_CODES with high drop forecasts.
        train_qrtr (list): List of training/validation quarters.
        forecast_qrtr (list): List of forecast quarters.

    Returns:
        pandas.DataFrame: DataFrame containing imputed forecasts.
    """

    # Calculate growth rates
    df_start_end_all["r"] = (
        df_start_end_all[forecast_qrtr[-1]] / df_start_end_all[train_qrtr[-1]]
    ) ** (1 / 8)
    df_start_end_all["beta"] = (
        df_start_end_all[forecast_qrtr[-1]] - df_start_end_all[train_qrtr[-1]]
    ) / 8

    # Linear imputation
    df_imputed_linear = pd.DataFrame(
        index=df_start_end_all.index, columns=forecast_qrtr
    )
    for q in range(len(df_imputed_linear.columns)):
        df_imputed_linear[df_imputed_linear.columns[q]] = (
            df_start_end_all[train_qrtr[-1]] + q * df_start_end_all["beta"]
        )

    # Geometric imputation
    df_imputed_geo = pd.DataFrame(index=df_start_end_all.index, columns=forecast_qrtr)
    for q in range(len(df_imputed_geo.columns)):
        df_imputed_geo[df_imputed_geo.columns[q]] = df_start_end_all[train_qrtr[-1]] * (
            (df_start_end_all["r"]) ** (q + 1)
        )

    # Model forecast
    df_model_forecast_actual = df_model_forecast.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_PRED"
    ).loc[df_start_end_all.index]

    # Average imputation for J_CODES with high drop forecasts
    df_imputed = (df_imputed_linear + df_imputed_geo + df_model_forecast_actual) / 3
    df_imputed_j_high_drop = df_imputed.loc[j_high_drop]

    # Average imputation for J_CODES with last quarter actuals copied
    df_imputed = (df_imputed_linear + df_imputed_geo) / 2
    df_imputed_j_lastQ_actual_copied = df_imputed.loc[j_lastQ_actual_copied]

    # Concatenate imputed dataframes
    df_imputed = pd.concat([df_imputed_j_lastQ_actual_copied, df_imputed_j_high_drop])

    # Restore the actual end point
    df_imputed = df_imputed.join(
        df_start_end_all.rename(
            {train_qrtr[-1]: "START", forecast_qrtr[-1]: "END"}, axis=1
        )
    )
    df_imputed[forecast_qrtr[-1]] = df_imputed["END"]
    return df_imputed[forecast_qrtr]


def identify_jcodes_high_growth(
    df_model_forecast: pd.DataFrame,
    forecast_qrtr: list,
    j_lastQ_actual_copied: list,
) -> list:
    """
    Identify J_CODES with high forecast growth.

    Args:
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        forecast_qrtr (list): List of forecast quarters.
        j_lastQ_actual_copied (list): List of J_CODES with last quarter actuals copied.

    Returns:
        list: List of J_CODES with high forecast growth.
    """

    # Pivot the forecast dataframe
    df_model_forecast_pivot = df_model_forecast.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_PRED"
    )

    # Filter out growing forecasts
    df_model_forecast_pivot = df_model_forecast_pivot[
        df_model_forecast_pivot[forecast_qrtr[-1]]
        > df_model_forecast_pivot[forecast_qrtr[0]]
    ]

    # Filter J_CODES with growth greater than 3%
    df_model_forecast_pivot_high_growth = df_model_forecast_pivot[
        (
            (
                df_model_forecast_pivot[forecast_qrtr[-1]]
                / df_model_forecast_pivot[forecast_qrtr[0]]
            )
            ** (1 / 8)
            - 1
            > 0.03
        )
    ]

    # Extract J_CODES with high growth excluding those with last quarter actuals copied
    j_high_growth = list(df_model_forecast_pivot_high_growth.index)
    return [c for c in j_high_growth if c not in j_lastQ_actual_copied]


def impute_high_growth(
    df_actual_train_val: pd.DataFrame,
    df_model_forecast: pd.DataFrame,
    j_high_growth: list,
    train_qrtr: list,
    forecast_qrtr: list,
) -> pd.DataFrame:
    """
    Impute high growth forecasts using actual and forecasted growth rates.

    Args:
        df_actual_train_val (pandas.DataFrame): DataFrame containing actual training/validation data.
        df_model_forecast (pandas.DataFrame): DataFrame containing model forecasts.
        j_high_growth (list): List of J_CODES with high growth forecasts.
        train_qrtr (list): List of training/validation quarters.
        forecast_qrtr (list): List of forecast quarters.

    Returns:
        pandas.DataFrame: DataFrame containing imputed high growth forecasts.
    """

    # Calculate actual growth rate over the last 2 years
    df_actual_high_growth = df_actual_train_val.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_TRUE"
    ).loc[j_high_growth]
    df_actual_high_growth["GROWTH"] = (
        df_actual_high_growth[train_qrtr[-1]] / df_actual_high_growth[train_qrtr[-8]]
    ) ** (1 / 8) - 1
    df_actual_high_growth = df_actual_high_growth[["GROWTH"]]
    df_actual_high_growth.columns = ["GROWTH_ACTUAL"]

    # Calculate forecast growth rate
    df_forecast_high_growth = df_model_forecast.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_PRED"
    ).loc[j_high_growth]
    df_forecast_high_growth["GROWTH"] = (
        df_forecast_high_growth[forecast_qrtr[-1]]
        / df_forecast_high_growth[forecast_qrtr[0]]
    ) ** (1 / 8) - 1
    df_forecast_high_growth = df_forecast_high_growth[["GROWTH"]]
    df_forecast_high_growth.columns = ["GROWTH_FORECAST"]

    # Aggregate and cap forecasted growth rate
    df_high_growth = df_actual_high_growth.join(df_forecast_high_growth)
    df_high_growth["GROWTH_FORECAST_CAPPED"] = df_high_growth["GROWTH_FORECAST"].map(
        lambda x: x if np.abs(x) <= 0.03 else 0.03
    )
    df_high_growth["GROWTH_AVG"] = df_high_growth[
        ["GROWTH_ACTUAL", "GROWTH_FORECAST_CAPPED"]
    ].mean(1)
    df_high_growth["GROWTH_AVG_CAPPED"] = df_high_growth["GROWTH_AVG"].map(
        lambda x: x if np.abs(x) <= 0.03 else 0.03
    )

    # Extract last training quarter ASP
    df_high_growth_lastQ_actual = df_actual_train_val.pivot(
        index="J_CODE", columns="QUARTER", values="ASP_TRUE"
    ).loc[j_high_growth][[train_qrtr[-1]]]
    df_high_growth_lastQ_actual = df_high_growth_lastQ_actual.loc[j_high_growth]

    # Adjust forecast by capped growth
    df_imputed_geo = pd.DataFrame(index=j_high_growth, columns=[forecast_qrtr])
    for q in range(len(df_imputed_geo.columns)):
        df_imputed_geo[df_imputed_geo.columns[q]] = df_high_growth_lastQ_actual[
            train_qrtr[-1]
        ] * ((df_high_growth["GROWTH_AVG_CAPPED"] + 1) ** (q + 1))

    # Combine forecasted ASP values and imputed geo values using a 25%-75% weighted average for high-growth J codes.
    df_imputed = (
        df_model_forecast.pivot(index="J_CODE", columns="QUARTER", values="ASP_PRED")
        .loc[j_high_growth]
        .values
        * 0.25
        + df_imputed_geo.loc[j_high_growth].values * 0.75
    )
    return pd.DataFrame(df_imputed, index=j_high_growth, columns=forecast_qrtr)
