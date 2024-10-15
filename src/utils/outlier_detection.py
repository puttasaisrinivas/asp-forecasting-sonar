import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def detect_outliers_exp_smoothing(
    input_data: pd.DataFrame,
    feature_name: str,
    smoothing_level: float,
    smoothing_trend: float,
    smoothing_seasonal: float,
) -> tuple:
    """
    Detect outliers using Exponential Smoothing.

    Parameters:
        input_data (DataFrame): Input DataFrame containing the time series data.
        feature_name (str): Name of the numerical feature to detect outlier.
        smoothing_level (float): Smoothing level param(alpha) for exponential smoothing
        smoothing_trend (float): Smoothing Trend param(beta)for exponential smoothing
        smoothing_seasonal (float): Smoothing seasonal param(gamma) for expo smoothing

    Returns (tuple):
        smoothed_values (DataFrame): Contains the smoothed values
        given by triple exponential smoothing algorithm
        exp_outlier_index (list): Contains list of indexes that are detected as outliers
        count_outliers_exp (int): Number of outliers for a J_CODE

    Notes:
        Get the best params of exponential smoothing using grid search
    """

    # Fit Exponential Smoothing model
    model = ExponentialSmoothing(
        input_data[feature_name],
        initialization_method="estimated",
        trend="add",
        seasonal="add",
        seasonal_periods=4,
    )

    results = model.fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
    )

    # Get smoothed values
    smoothed_values = results.fittedvalues

    # Calculate residuals (difference between observed and smoothed values)
    residuals = input_data[feature_name].astype(float) - smoothed_values

    # Calculate mean and standard deviation of residuals
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    # Set threshold for outlier detection (e.g., 3 standard deviations)
    threshold = 3 * residual_std

    # Detect outliers
    outliers = (residuals > residual_mean + threshold) | (
        residuals < residual_mean - threshold
    )

    # store the outlier index in a list
    exp_outlier_index = []

    # count no. of outliers gien by expo smoothing.
    count_outliers_exp = 0

    for i in range(len(outliers)):
        if outliers[i]:
            exp_outlier_index.append(i)
            count_outliers_exp += 1

    return smoothed_values, exp_outlier_index, count_outliers_exp


def detect_outliers_fb_prophet(
    input_data: pd.DataFrame, feature_name: str, date_col: str
) -> tuple:
    """
    Detect outliers using Facebook Prophet Algorithm

    Parameters:
        input_data (DataFrame): Input DataFrame containing the time series data.
        feature_name (str): Name of the numerical feature to detect outlier.
        date_col (str): Name of the Date column

    Returns:
        fb_outlier_index (list): Contains list of indexes that are detected as outliers
        count_outliers_fb (int): Number of outliers for each J_CODE
    """

    # Prepare data for Prophet
    input_data.reset_index(inplace=True)
    fb_data = input_data.rename(columns={date_col: "ds", feature_name: "y"})

    # Initialize Prophet model
    model = Prophet()

    # Fit the model
    model.fit(fb_data)

    # Make future predictions
    future = model.make_future_dataframe(
        periods=365
    )  # Adjust the number of periods as needed
    forecast = model.predict(future)

    # Identify outliers
    fb_data["forecast"] = forecast["yhat"]
    fb_data["residuals"] = fb_data["y"] - fb_data["forecast"]

    # Identify outliers in the dataset by calculating the residuals
    # Residuals that are greater than 3 standard deviations
    # away from the mean are flagged as outliers
    outliers = fb_data[abs(fb_data["residuals"]) > 3 * fb_data["residuals"].std()]

    # Extract the index of the outliers for future reference or analysis
    fb_outlier_index = outliers.index.to_list()

    # Count the number of outliers based on the 'forecast' column
    count_outliers_fb = outliers["forecast"].count()

    return fb_outlier_index, count_outliers_fb


# Function to calculate Mean Squared Error (MSE)
def calculate_mse(params: tuple, data: pd.DataFrame) -> float:
    """
    Calculate the average Mean Squared Error (MSE)
    using Exponential Smoothing with the given parameters.

    Parameters:
    - params (tuple): A tuple containing alpha, beta,
    and gamma values for Exponential Smoothing.
    - data (array-like): Time series data to be used
    for fitting the model and calculating MSE.

    Returns:
    - float: The average Mean Squared Error (MSE)
    over a 5-fold time series cross-validation.

    Notes:
    - This function utilizes Exponential Smoothing
    for time series forecasting.
    - It splits the data into training and validation
    sets using 5-fold time series cross-validation.
    - The Exponential Smoothing model is fitted on the
    training data and validated on the validation data.
    - The MSE is calculated for each validation set,
    and the average MSE is returned.
    """

    alpha, beta, gamma = params
    total_mse = 0
    tscv = TimeSeriesSplit(n_splits=2)  # 2-fold time series cross-validation

    for train_index, val_index in tscv.split(data):
        train_data, validation_data = data[train_index], data[val_index]

        # Fit the model
        model = ExponentialSmoothing(
            train_data, trend="add", seasonal="add", seasonal_periods=4
        )
        fitted_model = model.fit(
            smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma
        )

        # Make predictions on validation set
        y_pred = fitted_model.predict(start=0, end=len(validation_data) - 1)

        # Calculate MSE
        mse = mean_squared_error(validation_data, y_pred)
        total_mse += mse

    # Return average MSE
    return total_mse / tscv.n_splits


def grid_search_expo_smoothing(input_data: pd.DataFrame, feature_name: str) -> dict:
    """
    Gives the best smoothing parameters for
    Exponential Smoothing using Grid Search Method

    Parameters:
        input_data (DataFrame): Input DataFrame containing the time series data.
        feature_name (str): Name of the numerical feature to detect outlier.

    Returns:
        best_params (dict): Dicitonary contains the best smoothing parameters
    """

    # Define alpha, beta, and gamma values
    alpha_values = np.arange(0.2, 0.5, 0.05)
    beta_values = np.arange(0.2, 0.5, 0.05)
    gamma_values = np.arange(0.2, 0.5, 0.05)

    # Initialize variables to store best parameters and MSE
    best_params = None
    best_mse = float("inf")

    # Loop over all combinations of alpha, beta, and gamma values
    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                params = [alpha, beta, gamma]
                mse = calculate_mse(params, input_data[feature_name])

                # Update best parameters if MSE is lower
                if mse < best_mse:
                    best_mse = mse
                    best_params = params

    return best_params


def impute_outliers(
    master_data: pd.DataFrame,
    input_data: pd.DataFrame,
    prod_cd: str,
    date_col: str,
    feature_name: str,
    exp_outlier_index: list,
    fb_outlier_index: list,
    exp_forecast: pd.DataFrame,
    method: str,
    window_size: int = 0,
) -> pd.DataFrame:

    """
    Impute the Outlier Values with forecasted values
    given by Expo Smoothing Algorithm

    Parameters:
        master_data (DataFrame): DataFrame containing master_data of all the products
        input_data (DataFrame): Input DataFrame containing the time series data.
        prod_cd (str): Code of the Product (J_CODE)
        date_col (str): Date column of master_data
        feature_name (str): Name of the numerical feature to detect outlier.
        exp_outlier_index (list): Contains list of indexes that are detected
        as outliers by Exponential Smoothing Algorithm.
        fb_outlier_index (list): Contains list of indexes that are detected
        as outliers by FB Prophet Algorithm.
        exp_forecast (DataFrame): Contains smoothed values of triple expo smoothing
        method (str): Method to impute the outliers.
        For Moving Average pass 'MA', for Triple Exponential Smoothing pass 'TES'
        window_size (int, optional): Window size to calculate rolling average

    Returns:
        imputed_data (DataFrame): Returns the master data after imputing outlier values.


    Notes:
        - Outliers are identified based on both Exponential Smoothing
        and FB Prophet algorithms.
        - Imputed values are filled using either Moving Average
        or Triple Exponential Smoothing method.
        - The imputed feature name is suffixed with 'IMP'
        to indicate it's an imputed value.
        - Final imputed values are merged with the master data
        based on product code, year, and quarter.
    """

    # Create a copy of the input data to perform imputation
    # operations without altering the original data
    imputed_data = input_data.copy()

    # Convert the date columns to datetime format for both master_data and imputed_data
    master_data[date_col] = pd.to_datetime(master_data[date_col])
    imputed_data[date_col] = pd.to_datetime(imputed_data[date_col])

    # Add a new column 'J_CODE' to imputed_data, assigning the product code
    imputed_data["J_CODE"] = prod_cd

    # Extract the quarter from the date column and add it to imputed_data
    # This will help in aggregating data on a quarterly basis
    imputed_data["quarter"] = imputed_data[date_col].dt.to_period("Q")

    # Extract the year from the date column and add it to imputed_data
    # This will facilitate year-based analysis
    imputed_data["year"] = imputed_data[date_col].dt.year

    # Perform similar operations for master_data to ensure consistency
    master_data["quarter"] = master_data[date_col].dt.to_period("Q")
    master_data["year"] = master_data[date_col].dt.year

    # Iterate over the list of indices identified as outliers
    # in the exponential smoothing forecast data
    for i in exp_outlier_index:
        if i in fb_outlier_index:
            if method == "TES":
                # Impute outlier with forecasted value from Exponential Smoothing
                imputed_value = exp_forecast.at[i]
                imputed_data.at[i, feature_name] = imputed_value
            elif method == "MA":
                # Impute outlier with rolling mean
                rolling_mean = (
                    imputed_data[feature_name]
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                    .shift()
                )
                imputed_data.at[i, feature_name] = rolling_mean[i]

    imputed_feaure_name = feature_name + "IMP"
    imputed_data = imputed_data.rename(columns={feature_name: imputed_feaure_name})

    # Merge imputed data with master data
    merged_df = pd.merge(
        master_data,
        imputed_data[[imputed_feaure_name, "year", "quarter", "J_CODE"]],
        on=["year", "quarter", "J_CODE"],
        how="left",
    )
    merged_df[feature_name] = merged_df[imputed_feaure_name].fillna(
        merged_df[feature_name]
    )
    merged_df.drop(columns=[imputed_feaure_name, "year", "quarter"], inplace=True)

    return merged_df


def outlier_detection(
    master_data: pd.DataFrame,
    lst_prioritized_products: pd.DataFrame,
    feature_name: str,
    data_freq: str,
    impute_method: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Detect outliers for all the products, remove the outlier values,
    and impute missing values.

    Parameters:
        master_data (pd.DataFrame): Master data containing relevant features.
        lst_prioritized_products (DataFrame): List of prioritized products.
        feature_name (str): Name of the feature on which outlier detection is needed.
        data_freq (str): Frequency of data (Monthly, Quarterly).
        For monthly pass 'ME' and for Quarterly pass 'QE'.
        impute_method (str): Method to impute the outliers.
        For Moving Average pass 'MA', for Triple Exponential Smoothing pass 'TES'.
        logger (logging.Logger): Logger object for logging messages.

    Returns:
        pd.DataFrame: Master Data with updated ASP_PRICE
        after removing outliers and imputing the missing values.

    Notes:
        - Outlier detection is performed using Exponential Smoothing and FB Prophet.
        - Missing values are handled by filling
        with the rolling mean and forward/backward fill.
        - The method to impute outliers depends on the specified impute_method.
    """

    # Create a child logger for detecting outliers
    child_logger = logger.getChild("Detect Outliers")
    j_code_list = lst_prioritized_products
    date_col = "CAL_DT"

    # Iterate over each product code in the prioritized list
    for prod_cd in j_code_list:
        data = master_data[["J_CODE", date_col, feature_name]].drop_duplicates()
        data = data[data["J_CODE"] == prod_cd][[date_col, feature_name]]
        data[date_col] = pd.to_datetime(data[date_col])

        # Determine the rolling window size based on data frequency
        if data_freq == "QE":
            window_size = 4
        else:
            window_size = 3

        # Creating resampled data
        ts = data[[date_col, feature_name]]
        ts.set_index(date_col, inplace=True)
        resampled_data = ts.resample(data_freq).first()
        resampled_data.reset_index(inplace=True)

        # Missing values handling
        resampled_data[feature_name] = resampled_data[feature_name].replace(
            "null", np.nan
        )

        # Fill the missing values with rolling mean of window 4
        resampled_data[feature_name] = resampled_data[feature_name].fillna(
            resampled_data[feature_name]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .shift()
        )

        # Fill the remaining missing values first with
        # forward fill and then backward fill
        resampled_data[feature_name] = resampled_data[feature_name].ffill()
        resampled_data[feature_name] = resampled_data[feature_name].bfill()

        # Convert feature_name column to float (if necessary)
        resampled_data[feature_name] = resampled_data[feature_name].astype(float)

        # Get the best parameters from grid search
        best_params = grid_search_expo_smoothing(resampled_data, feature_name)
        # Outlier detection using Exponential Smoothing
        (
            smoothed_values_exp,
            exp_outlier_index,
            count_outliers_exp,
        ) = detect_outliers_exp_smoothing(
            resampled_data, feature_name, best_params[0], best_params[1], best_params[2]
        )

        # If Exponential smoothing is giving any outlier for a
        # J_CODE then we will be running FB Prophet to confirm
        if count_outliers_exp > 0:
            # Outlier Detection using FB Prophet
            fb_outlier_index, count_outliers_fb = detect_outliers_fb_prophet(
                resampled_data, feature_name, date_col
            )

            # If all the points are captured as outliers
            # by FB Prophet then don't impute the data.
            if count_outliers_fb > 0 and resampled_data.shape[0] != count_outliers_fb:
                child_logger.info(f"Outlier Detected for {prod_cd}")
                child_logger.info(f"exp_outlier_index: {exp_outlier_index} ")
                child_logger.info(f"fb_outlier_index: {fb_outlier_index}")
                # MA represents Imputing with Moving Average
                # TES represnts imputing with Triple Exponential smoothing
                if impute_method == "MA":
                    master_data = impute_outliers(
                        master_data,
                        resampled_data,
                        prod_cd,
                        date_col,
                        feature_name,
                        exp_outlier_index,
                        fb_outlier_index,
                        smoothed_values_exp,
                        impute_method,
                        window_size,
                    )
                elif impute_method == "TES":
                    master_data = impute_outliers(
                        master_data,
                        resampled_data,
                        prod_cd,
                        date_col,
                        feature_name,
                        exp_outlier_index,
                        fb_outlier_index,
                        smoothed_values_exp,
                        impute_method,
                    )

    return master_data
