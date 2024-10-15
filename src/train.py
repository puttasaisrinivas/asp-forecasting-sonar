import logging
import yaml
import pandas as pd
from models.arima import train_and_predict_with_arima
from transform import (
    get_static_features,
    transform_data,
    get_market_event_features_for_generic,
    update_event_features_for_non_generic,
)
from models.xgb import train_and_predict_residuals_xgb, predict_with_trained_xgb
from utils.custom_model import load_model_for_inference


def forecast_asp(
    master_data_processed: pd.DataFrame,
    market_event_data_processed: pd.DataFrame,
    operation_mode: str,
    model_alias: str,
    config: yaml,
    config_dict: dict,
    test_months: list,
    generic_flag: bool,
    train_quarters: list,
    mlflow_obj: object,
    categorical_cols: list,
    model_records: list,
    registry_records: list,
    data_drift_records: list,
    catalog_name: str,
    run_id: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Forecast ASP using ARIMA and XGBoost models.

    Args:
        master_data_processed (pd.DataFrame): Processed master data.
        market_event_data_processed (pd.DataFrame): Processed market event data.
        operation_mode (str): retraining or inference.
        model_alias (str): alias of the model (challenger/champion etc.)
        config (yaml): A Config object initialized with the data from the YAML file
        config_dict (Dict[str, any]): Configuration parameters.
        test_months (List[str]): List of months to be used for testing.
        generic_flag (bool): Flag indicating if generic features are used.
        train_quarters (List[str]): List of quarters used for training.
        mlflow_obj (MlflowClient): MLflow client object for tracking experiments.
        categorical_cols (List[str]): List of categorical columns.
        model_records (list): List to hold the model records as dictionaries.
        registry_records (list): List to hold the registry records as dictionaries.
        data_drift_records (list): List to store baseline dataset details
        of each model as a dictionary.
        catalog_name (str): Name of Unity Catalog.
        logger (logging.Logger): Logger object for logging information.

    Returns:
        pd.DataFrame: DataFrame containing ASP forecasts.

    Notes:
        - This function forecasts ASP using ARIMA and XGBoost models.
        - It trains ARIMA model to predict the forecast
        and prepares features for XGBoost.
        - XGBoost is then trained to forecast the residuals.
    """

    child_logger = logger.getChild("FORECAST_ASP_TRAIN")
    child_logger.info("Training arima..")
    # Step 1: Train ARIMA model and predict forecast
    arima_predicted_data = train_and_predict_with_arima(
        processed_master_data=master_data_processed,
        forecast_months=config_dict["forecast_months"],
        target_variable=config_dict["target_variable"],
        time_series_columns=config_dict["time_series_columns"],
        logger=logger,
    )
        
    child_logger.info("Trained arima and predicted the forecast succesfully!")

    child_logger.info(
        "Getting residuals and other exogenous features ready for step-2.."
    )
    # Step 2: Prepare static and market event features for XGBoost
    static_feature_data = get_static_features(
        processed_master_data=master_data_processed,
        generic_flag=generic_flag,
        forecast_months=config_dict["forecast_months"],
        static_columns=config_dict["static_columns"],
    )

    scaler_obj, transformed_data_scaled = transform_data(
        predicted_arima_forecast=arima_predicted_data,
        time_series_columns=config_dict["time_series_columns"],
        test_months=test_months,
    )

    if generic_flag:
        market_event_feature_data = get_market_event_features_for_generic(
            processed_master_data=master_data_processed,
            time_cap=config_dict["time_cap"],
        )
    else:
        market_event_feature_data = pd.DataFrame()
        static_feature_data = update_event_features_for_non_generic(
            static_feature_data=static_feature_data,
            market_event_data=market_event_data_processed,
            train_data_end_quarter=max(train_quarters),
        )

    if operation_mode == "retraining":
        """
        Step 1: Train and Predict ASP using ARIMA
        Step 2: Train and Predict residuals using XGBoost
        Step 3: store the model details in registy_records
        list for registering to mlflow.
        """
        child_logger.info("The Data is ready to be trained! ")
        child_logger.info("Training xg-boost to forecast the residuals..")
        # Step 3: Train XGBoost to forecast residuals
        predictions_scaled = train_and_predict_residuals_xgb(
            transformed_data=transformed_data_scaled,
            market_event_feature_data=market_event_feature_data,
            static_feature_data=static_feature_data,
            master_data_processed=master_data_processed,
            arima_predicted_data=arima_predicted_data,
            config_dict=config_dict,
            generic_flag=generic_flag,
            test_months=test_months,
            scaler=scaler_obj,
            categorical_columns=categorical_cols,
            registry_records=registry_records,
            run_id=run_id,
            logger=child_logger,
        )

        model_version = None

    elif operation_mode == "inference":
        """
        Step 1: Train and Predict ASP using ARIMA
        Step 2: Load the model from mlflow model registry that is tagged
        as champion for champion pipeline and challeneger for challeneger pipeline.
        Step 3: Predict residuals using XGB from loaded Model.
        """
        child_logger.info("Loading Logged models from mlflow model registry")
        (
            trained_model_scaled,
            feature_names,
            feature_dtypes,
            champion_baseline_path,
            model_version,
        ) = load_model_for_inference(
            generic_flag=generic_flag,
            mlflow_obj=mlflow_obj,
            config=config,
            catalog_name=catalog_name,
            model_alias=model_alias,
            logger=child_logger,
        )

        predictions_scaled = predict_with_trained_xgb(
            transformed_data=transformed_data_scaled,
            market_event_feature_data=market_event_feature_data,
            static_feature_data=static_feature_data,
            master_data_processed=master_data_processed,
            arima_predicted_data=arima_predicted_data,
            config_dict=config_dict,
            generic_flag=generic_flag,
            test_months=test_months,
            scaler=scaler_obj,
            categorical_columns=categorical_cols,
            feature_names=feature_names,
            feature_dtypes=feature_dtypes,
            model=trained_model_scaled,
            champion_baseline_dataset_path=champion_baseline_path,
            data_drift_records=data_drift_records,
            logger=child_logger,
        )

    predictions_scaled = predictions_scaled[
        ["J_CODE", "TIME", "ASP_FORECASTED_MEAN"]
    ].rename(columns={"ASP_FORECASTED_MEAN": "ASP_PRED_SCALED", "TIME": "QUARTER"})

    return predictions_scaled, model_version
