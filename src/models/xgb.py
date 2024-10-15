import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from utils import constants
from predict import get_xgb_predictions, aggregate_predictions, retransform_predictions
import shutil
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler


def prepare_sliding_window_data_for_xgboost(
    transformed_data: pd.DataFrame,
    market_event_feature_data: pd.DataFrame,
    static_feature_data: pd.DataFrame,
    sliding_window_size: int,
    generic_flag: bool,
    time_series: str,
) -> pd.DataFrame:
    """
    Prepare sliding window data for XGBoost model training.

    Args:
        transformed_data (pd.DataFrame): Transformed data containing
        time series information.
        market_event_feature_data (pd.DataFrame): Market event feature data.
        static_feature_data (pd.DataFrame): Static feature data.
        sliding_window_size (int): Size of the sliding window.
        generic_flag (bool): A flag indicating whether generic features should be used.
        time_series (str): Name of the time series column.

    Returns:
        pd.DataFrame: Prepared sliding window feature data.

    Notes:
        - The function prepares sliding window data for XGBoost model training.
        - It joins static and market event feature data to the sliding window data.
        - Categorical columns are converted to categorical data types.
    """

    # Filter transformed data based on time series
    transformed_data = transformed_data[
        transformed_data["TIME_SERIES"] == time_series
    ].drop(columns=["TIME_SERIES"])

    month_cols = [c for c in transformed_data.columns if "-" in c]
    i = 0
    data = []

    # Iterate through sliding window
    for i in range(len(month_cols) - sliding_window_size):
        start_month = month_cols[i]
        end_month = month_cols[i + sliding_window_size - 1]
        target_month = month_cols[i + sliding_window_size]

        # Prepare X and Y data
        x = pd.DataFrame(
            transformed_data.loc[:, start_month:end_month].values,
            index=transformed_data.J_CODE,
            columns=[f"{str(c)}_{time_series}" for c in range(sliding_window_size)],
        )
        y = pd.DataFrame(
            transformed_data.loc[:, target_month].values,
            index=transformed_data.J_CODE,
            columns=[time_series],
        )
        xy = x.join(y)
        xy["WINDOW_START"] = start_month
        xy["WINDOW_END"] = end_month
        xy["TARGET_MONTH"] = target_month
        data.append(xy)

    data = pd.concat(data)
    data = (
        data.reset_index()
        .sort_values(by=["J_CODE", "TARGET_MONTH"])
        .set_index(["J_CODE", "WINDOW_START", "WINDOW_END", "TARGET_MONTH"])
    )

    # Join static feature data
    if generic_flag:
        data_all = data.join(
            static_feature_data[
                ["J_CODE", "THERA_CLS_DSCR", "DISEASE_ST", "PROD_HIER1"]
            ].set_index("J_CODE")
        )
        market_event_feature_data = market_event_feature_data.rename(
            columns={"ASP_MTH": "TARGET_MONTH"}
        )
        data_all = data_all.join(
            market_event_feature_data.set_index(["J_CODE", "TARGET_MONTH"])
        )
    else:
        data_all = data.join(
            static_feature_data[
                [
                    "J_CODE",
                    "THERA_CLS_DSCR",
                    "MSPN_DSG_FRM",
                    "MSPN_ROA",
                    "USC5_DSCR",
                    "DISEASE_ST",
                    "PROD_HIER1",
                    "TIME_SINCE_LAST_COMP_LAUNCH_TRAIN_FLAG",
                    "TIME_SINCE_LAST_LOE_TRAIN_FLAG",
                ]
            ].set_index("J_CODE")
        )

    # Convert data types
    categorical_columns = list(static_feature_data.columns[2:])
    numerical_col = list(set(data_all.columns) - set(categorical_columns))
    data_all[numerical_col] = data_all[numerical_col].fillna(np.nan)
    data_all[numerical_col] = data_all[numerical_col].astype(float)

    # Prepare window-wise feature data
    window_wise_feature_data = data_all.copy()
    window_wise_feature_data = window_wise_feature_data.reset_index()
    window_wise_feature_data.set_index(["J_CODE", "TARGET_MONTH"], inplace=True)
    window_wise_feature_data = window_wise_feature_data.T.drop_duplicates().T
    window_wise_feature_data[categorical_columns] = window_wise_feature_data[
        categorical_columns
    ].astype("category")
    window_wise_feature_data["MONTH_FEATURE"] = (
        window_wise_feature_data.reset_index()["TARGET_MONTH"]
        .map(lambda x: int(str(x).split("-")[-1]))
        .values
    )
    numerical_col = list(
        set(window_wise_feature_data.columns[2:]) - set(categorical_columns)
    )
    window_wise_feature_data[numerical_col] = window_wise_feature_data[
        numerical_col
    ].astype(float)

    return window_wise_feature_data


def get_train_test_data(
    window_wise_feature_data: pd.DataFrame,
    test_months: list,
    time_series_columns: list,
) -> tuple:
    """
    Get train and test data for model training and evaluation.

    Args:
        window_wise_feature_data (pd.DataFrame): Window-wise feature data.
        test_months (list): List of months to be used for testing.
        time_series_columns (list): List of time series columns.

    Returns:
        tuple: A tuple containing X_train and y_train DataFrames.

    Raises:
        ValueError: If the input data is not provided or is empty.

    Notes:
        - The function filters the window-wise feature data based on test months.
        - It separates features and target variables for
        both training and testing datasets.
    """

    exclude_columns = [
        "J_CODE",
        "WINDOW_START",
        "WINDOW_END",
        "TARGET_MONTH",
    ] + time_series_columns

    # Filter training data
    window_wise_feature_data_train = window_wise_feature_data[
        ~window_wise_feature_data.index.get_level_values("TARGET_MONTH").isin(
            test_months
        )
    ]

    # Separate features and target variables for training data
    X_train = window_wise_feature_data_train.drop(
        columns=[
            c for c in window_wise_feature_data_train.columns if c in exclude_columns
        ]
    )
    y_train = window_wise_feature_data_train[time_series_columns]

    return X_train, y_train


def map_with_default(value: str, custom_mapping: dict) -> int:
    """
    Maps a string value to an integer based on a custom mapping dictionary.
    Returns a default value of -1 if the string is not found in the dictionary.

    Args:
        value (str): The string value to map.
        custom_mapping (dict): The dictionary containing
        the custom mappings from string to integer.

    Returns:
        int: The mapped integer value, or -1 if the string
        is not found in the custom mapping.
    """

    return custom_mapping.get(value, -1)  # default value -1 if not found


def train_xgb(X_train: pd.DataFrame, y_train: pd.DataFrame) -> XGBRegressor:
    """
    Train an XGBoost model using the provided training data.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.DataFrame): Target variables for training.

    Returns:
        XGBRegressor: Trained XGBoost model.

    Raises:
        ValueError: If the input data is not provided or is invalid.
    """

    # Filter out rows with NaN target values
    idx = y_train.isna().sum(1)
    idx = idx[idx == 0].index
    X_train_filtered = X_train.loc[idx]
    y_train_filtered = y_train.loc[idx]

    random_seed = 42

    # Train XGBoost model
    model = XGBRegressor(enable_categorical=True, seed=random_seed)
    default_xgb_params = model.get_params()
    model.fit(X_train_filtered, y_train_filtered)

    # Return the trained model, scaler (if used), and default parameters
    return model, default_xgb_params


def one_hot_encode_with_others(
    dataframe: pd.DataFrame, column_name: str, categories: list, prefix: str
) -> pd.DataFrame:
    """
    Perform one-hot encoding for a specified column and
    add missing categories as "others".

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to be one-hot encoded.
        categories (List[str]): List of categories for the column.
        prefix (str): Prefix for new column names after encoding.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded column.

    Notes:
        - This function performs one-hot encoding for the specified column.
        - It adds missing categories as "others" and
        uses the provided prefix for new column names.
    """

    # Create one-hot encoding for the specified column
    one_hot_encoded = pd.get_dummies(dataframe[column_name], prefix=prefix)

    # Add missing categories as "others"
    for category in categories:
        category = "{}_".format(
            prefix,
        )
        if category not in one_hot_encoded.columns:
            one_hot_encoded[category] = 0
        one_hot_encoded[category] = one_hot_encoded[category].astype("int32")
    return one_hot_encoded


def convert_into_categorical(
    dataframe: pd.DataFrame, cat_cols: list[str], prefix: dict, logger: logging
) -> pd.DataFrame:
    """
    Convert specified columns into categorical and perform one-hot encoding.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        cat_cols (List[str]): List of columns to be converted into categorical.
        prefix (Dict[str, str]): Prefix dictionary for new column names after encoding.
        logger (logging.Logger): Logger object for logging information.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted into categorical.

    Notes:
        - This function converts the specified columns into categorical variables.
        - It performs one-hot encoding and adds a prefix to the new column names.
        - Original columns are dropped from the DataFrame.
    """

    child_logger = logger.getChild("CONVERT_INTO_CATEGORICAL")
    child_logger.info(
        "converting following columns into categorical:{}".format(cat_cols)
    )

    # One-hot encode the column with a unique category
    for col in cat_cols:
        p_ = prefix[col]
        list_of_categories = dataframe[col].unique().tolist()
        list_of_categories += ["{}_others".format(col)]
        encoded_column = one_hot_encode_with_others(
            dataframe, col, list_of_categories, p_
        )
        dataframe = pd.concat([dataframe, encoded_column], axis=1)
    # Drop columns
    dataframe.drop(cat_cols, axis=1, inplace=True)
    return dataframe


def train_and_predict_residuals_xgb(
    transformed_data: pd.DataFrame,
    market_event_feature_data: pd.DataFrame,
    static_feature_data: pd.DataFrame,
    master_data_processed: pd.DataFrame,
    arima_predicted_data: pd.DataFrame,
    config_dict: dict,
    generic_flag: bool,
    test_months: list,
    scaler: StandardScaler,
    categorical_columns: list,
    registry_records: list,
    run_id: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Train an XGBoost model and predict residuals.

    Args:
        transformed_data (pd.DataFrame): Transformed time series data.
        market_event_feature_data (pd.DataFrame): Market event feature data.
        static_feature_data (pd.DataFrame): Static feature data.
        master_data_processed (pd.DataFrame): Processed master data.
        arima_predicted_data (pd.DataFrame): ARIMA predicted data.
        config_dict (dict): Configuration dictionary.
        generic_flag (bool): Flag indicating generic or non-generic products.
        test_months (list): List of months for testing.
        scaler (StandardScaler): Scaler object for scaling features.
        categorical_columns (list): list of categorical cols that used for training xgb.
        registry_records (list): list to hold the model details to register in mlflow.
        run_id (str): Unique identifier for the current model training run.
        logger (logging.Logger): Logger object for recording process info and errors.

    Returns:
        pd.DataFrame: Retransformed predictions.

    Notes:
        - This function trains an XGBoost model, predicts residuals,
        aggregates predictions, retransforms them, and returns the final predictions.
    """

    child_logger = logger.getChild("TRAIN_RESIDUALS_USING_XGB")
    child_logger.info("Preparing sliding window data for Xgboost")
    child_logger.info(
        "Flags - generic:{}".format(generic_flag)
    )

    # Setting up root dir
    root_folder = (
            config_dict["DATA"]["files_location"] 
            + constants.MODELS_FOLDER 
        )

    if generic_flag:
        root_folder = root_folder + "generic/"
    else:
        root_folder = root_folder + "non-generic/"

    if not os.path.exists(root_folder):
        child_logger.info("setting up Necessary folders to save model artifacts")
        os.makedirs(root_folder)
    else:
        try:
            shutil.rmtree(root_folder)
        except Exception as e:
            child_logger.info(f"An Error occured while cleaning up directories:{e}")
            raise Exception("Error ocurred while cleaning up the directories")
        # Create empty directory
        os.makedirs(root_folder)

    # Prepare sliding window data for XGBoost
    window_wise_feature_data = prepare_sliding_window_data_for_xgboost(
        transformed_data=transformed_data,
        market_event_feature_data=market_event_feature_data,
        static_feature_data=static_feature_data,
        sliding_window_size=config_dict["window_size"],
        generic_flag=generic_flag,
        time_series=config_dict["time_series"],
    )

    # To avoid duplicate columns while creating one-hot encoding
    prefix = {
        "THERA_CLS_DSCR": "THD",
        "DISEASE_ST": "DS",
        "PROD_HIER1": "PH",
        "MSPN_ROA": "MSPNROA",
        "MSPN_DSG_FRM": "MSPNDSG",
        "USC5_DSCR": "USC",
    }

    child_logger.info("Converting categorical columns into One-Hot Encoding")
    window_wise_feature_data = convert_into_categorical(
        window_wise_feature_data, categorical_columns, prefix, child_logger
    )
    child_logger.info(
        "Dataframe shape after converting categories to one-hot encoding:{}".format(
            window_wise_feature_data.shape
        )
    )

    # Get train and test data
    X_train, y_train = get_train_test_data(
        window_wise_feature_data=window_wise_feature_data,
        test_months=test_months,
        time_series_columns=config_dict["time_series_columns"],
    )

    X_train.to_csv(root_folder + "baseline_dataset_features.csv", sep=",", index=True)
    y_train.to_csv(root_folder + "baseline_dataset_target.csv", sep=",", index=True)

    # Convert categorical columns into numerical columns
    categorical_columns = X_train.select_dtypes(include=["category"]).columns
    for cc in categorical_columns:
        X_train[cc] = X_train[cc].astype(int)

    # Saving x_train, and y_train dataframe locally
    child_logger.info(
        "Training Dataframe Shape:{}, {}".format(X_train.shape, y_train.shape)
    )
    # Train XGBoost model
    child_logger.info("#### Training XGBoost Model ####")
    model, default_xgb_params = train_xgb(X_train=X_train, y_train=y_train)
    child_logger.info("Model Type:{}".format(type(model)))

    # Infere signature
    model_sign = infer_signature(X_train, y_train)

    # Get XGBoost predictions
    child_logger.info("Getting predictions from the Trained model")
    xgb_predicted_data = get_xgb_predictions(
        window_wise_feature_data=window_wise_feature_data,
        model=model,
        window_size=config_dict["window_size"],
        processed_master_data=master_data_processed,
        X_train=X_train,
        y_train=y_train,
        test_months=test_months,
        forecast_months=config_dict["DATA"]["forecast_months"],
        time_series_columns=config_dict["time_series_columns"],
        logger=child_logger,
    )
    xgb_predicted_data.to_csv(
        root_folder + "xgb_predicted_data.csv", sep=",", index=False
    )

    # Aggregate XGBoost predictions with ARIMA predictions
    predictions_aggregated = aggregate_predictions(
        xgb_pred_data=xgb_predicted_data,
        arima_predicted_data=arima_predicted_data,
        scaler=scaler,
    )
    predictions_aggregated.to_csv(
        root_folder + "predictions_aggregated.csv", sep=",", index=False
    )

    # Retransform predictions
    predictions_retransformed = retransform_predictions(
        aggregated_predictions=predictions_aggregated,
        processed_master_data=master_data_processed,
        generic_flag=generic_flag,
    )
    predictions_retransformed.to_csv(
        root_folder + "predictions.csv", sep=",", index=False
    )

    registry_records.append(
        {
            "trained_model": model,
            "generic_flag": generic_flag,
            "scaler": scaler,
            "model_signature": model_sign,
            "xgb_params": default_xgb_params,
        }
    )
    return predictions_retransformed


def predict_with_trained_xgb(
    transformed_data: pd.DataFrame,
    market_event_feature_data: pd.DataFrame,
    static_feature_data: pd.DataFrame,
    master_data_processed: pd.DataFrame,
    arima_predicted_data: pd.DataFrame,
    config_dict: dict,
    generic_flag: bool,
    test_months: list,
    scaler: StandardScaler,
    categorical_columns: list,
    feature_names: list,
    feature_dtypes: dict,
    model: XGBRegressor,
    champion_baseline_dataset_path: str,
    data_drift_records: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Use a pre-trained XGBoost model to make predictions on new data.

    Args:
        transformed_data (pd.DataFrame): Transformed time series data.
        (New data to predict residuals for)
        market_event_feature_data (pd.DataFrame): Market event feature data.
        static_feature_data (pd.DataFrame): Static feature data.
        master_data_processed (pd.DataFrame): Processed master data.
        arima_predicted_data (pd.DataFrame): ARIMA predicted data.
        config_dict (dict): Configuration dictionary.
        generic_flag (bool): Flag indicating generic or non-generic products.
        test_months (list): List of months for which validations are to be done.
        scaler (StandardScaler): Scaler object for scaling features.
        categorical_columns (list): list of categorical cols that used for training xgb.
        feature_names (list): list of features logged for training xgb.
        feature_dtypes (dict): dictionary of feature data types.
        model (XGBRegressor): loaded model from mlflow model registry.
        champion_baseline_dataset_path (str): baseline dataset path of champion model.
        data_drift_records (list): List with records of data drift as disctionary.
        logger (logging.Logger): Logger to log information.
    Returns:
        pd.DataFrame: Retransformed predictions.

    Notes:
        - This function predicts residuals of ARIMA using trained model,
        aggregates predictions, retransforms them, and returns the final predictions.
    """

    child_logger = logger.getChild("PREDICT_ASP_USING_TRAINED_XGB")
    child_logger.info("Preparing sliding window data for Xgboost")
    child_logger.info(
        "Flags - generic:{}".format(generic_flag)
    )

    # setting up root dir
    root_folder = (
            config_dict["DATA"]["files_location"] 
            + constants.MODELS_FOLDER 
        )

    if generic_flag:
        root_folder = root_folder + "generic/"
    else:
        root_folder = root_folder + "non-generic/"

    if not os.path.exists(root_folder):
        child_logger.info("setting up Necessary folders to save model artifacts")
        os.makedirs(root_folder)
    else:
        try:
            shutil.rmtree(root_folder)
        except Exception as e:
            child_logger.info(f"An Error occured while cleaning up directories:{e}")
            raise Exception("Error ocurred while cleaning up the directories")
        # Create empty directory
        os.makedirs(root_folder)

    # Prepare sliding window data for XGBoost
    window_wise_feature_data = prepare_sliding_window_data_for_xgboost(
        transformed_data=transformed_data,
        market_event_feature_data=market_event_feature_data,
        static_feature_data=static_feature_data,
        sliding_window_size=config_dict["window_size"],
        generic_flag=generic_flag,
        time_series=config_dict["time_series"],
    )

    # To avoid duplicate columns while creating one-hot encoding
    prefix = {
        "THERA_CLS_DSCR": "THD",
        "DISEASE_ST": "DS",
        "PROD_HIER1": "PH",
        "MSPN_ROA": "MSPNROA",
        "MSPN_DSG_FRM": "MSPNDSG",
        "USC5_DSCR": "USC",
    }

    child_logger.info("Converting categorical columns into One-Hot Encoding")
    window_wise_feature_data = convert_into_categorical(
        window_wise_feature_data, categorical_columns, prefix, child_logger
    )
    child_logger.info(
        "Dataframe shape after converting categories to one-hot encoding:{}".format(
            window_wise_feature_data.shape
        )
    )

    child_logger.info("window wise feature data columns before converting the dtypes")
    for column, dtype in window_wise_feature_data.dtypes.items():
        # Convert categorical columns into numerical columns
        if dtype == "category":
            window_wise_feature_data[column] = pd.to_numeric(
                window_wise_feature_data[column], errors="coerce"
            )

    # Get train and test data
    X_train, y_train = get_train_test_data(
        window_wise_feature_data=window_wise_feature_data,
        test_months=test_months,
        time_series_columns=config_dict["time_series_columns"],
    )

    X_train.to_csv(root_folder + "baseline_dataset_features.csv", sep=",", index=True)
    y_train.to_csv(root_folder + "baseline_dataset_target.csv", sep=",", index=True)

    # Convert categorical columns into numerical columns
    categorical_columns = X_train.select_dtypes(include=["category"]).columns
    for cc in categorical_columns:
        X_train[cc] = X_train[cc].astype("int32")

    # Saving x_train, and y_train dataframe locally
    child_logger.info(
        "Training Dataframe Shape:{}, {}".format(X_train.shape, y_train.shape)
    )

    model_required_columns = feature_dtypes.keys()
    already_present_columns = window_wise_feature_data.columns.tolist()
    for model_col in model_required_columns:
        # Get the dtype the model requires
        dtype = feature_dtypes[model_col]
        dtype = str(dtype).lower()
        # Check if column required for model present in the window wise dataset
        if model_col not in already_present_columns:
            window_wise_feature_data[model_col] = 0

        # Convert the columns into required model type
        if "integer" in dtype:
            window_wise_feature_data[model_col] = window_wise_feature_data[
                model_col
            ].astype("int32")
        elif "double" in dtype:
            window_wise_feature_data[model_col] = window_wise_feature_data[
                model_col
            ].astype("float64")
        elif "long" in dtype:
            window_wise_feature_data[model_col] = window_wise_feature_data[
                model_col
            ].astype("int64")

    # Filter out model required columns only
    window_wise_feature_data = window_wise_feature_data[model_required_columns]

    if champion_baseline_dataset_path is not None:
        data_drift_records.append(
            {
                "current_baseline_path": window_wise_feature_data,
                "champion_baseline_path": champion_baseline_dataset_path,
                "root_folder": root_folder,
            }
        )

    # Get XGBoost predictions
    child_logger.info("Getting predictions from the Trained model")
    xgb_predicted_data = get_xgb_predictions(
        window_wise_feature_data=window_wise_feature_data,
        model=model,
        window_size=config_dict["window_size"],
        processed_master_data=master_data_processed,
        X_train=X_train,
        y_train=y_train,
        test_months=test_months,
        forecast_months=config_dict["DATA"]["forecast_months"],
        time_series_columns=config_dict["time_series_columns"],
        logger=child_logger,
    )
    xgb_predicted_data.to_csv(
        root_folder + "xgb_predicted_data.csv", sep=",", index=False
    )

    # Aggregate XGBoost predictions with ARIMA predictions
    predictions_aggregated = aggregate_predictions(
        xgb_pred_data=xgb_predicted_data,
        arima_predicted_data=arima_predicted_data,
        scaler=scaler,
    )
    predictions_aggregated.to_csv(
        root_folder + "predictions_aggregated.csv", sep=",", index=False
    )

    # Retransform predictions
    predictions_retransformed = retransform_predictions(
        aggregated_predictions=predictions_aggregated,
        processed_master_data=master_data_processed,
        generic_flag=generic_flag,
    )
    predictions_retransformed.to_csv(
        root_folder + "predictions.csv", sep=",", index=False
    )

    return predictions_retransformed
