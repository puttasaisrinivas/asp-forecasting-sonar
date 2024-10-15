import pandas as pd
from utils import constants
from utils.helpers import get_config_from_file, read_data, get_config
from utils.logger import setup_logger
from utils.dir import cleanup_dir, setup_dir
from extract import data_process
from config import ConfigDetails
from processing.process import create_train_test_month_quarters
from processing.post_processing import post_process
from train import forecast_asp
from utils.track import MLFlowManager
from utils.dqm import dqm_checks
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from utils.outlier_detection import outlier_detection
from datetime import datetime
from pyspark.sql.functions import col, lit, upper, min as min_, round
from pyspark.sql.types import StructType, StructField, StringType
from utils.custom_model import save_model_mlflow
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("asp-forecasting").getOrCreate()


def main():

    # load configuration file
    """Load configuration File"""
    config = get_config_from_file(constants.CONFIG_FOLDER_PATH + "general.yaml")

    """Setup and clean dir"""
    cleanup_dir(config)
    setup_dir(config)

    # get the paramaters from the pipeline
    params = sys.argv[1:]
    parsed_params = {}
    for param in params:
        key, value = param.split(":")
        parsed_params[key] = value
    catalog_name = parsed_params.get("catalog_name", "default_catalog")
    storage_account = parsed_params.get("storage_account", "default_storage")

    """ Logger setup """
    main_logger, log_file = setup_logger(
        constants.LOG_FILENAME,
        config["DATA"]["files_location"],
        constants.CODE_VERSION + ".MAIN",
    )

    """ Create MLFlow instance to track program artifacts """
    main_logger.info("Initializing MLFlow")
    mlflow_obj = MLFlowManager(
        experiment_path=config["MLFLOW_CONFIGS"]["EXPERIMENT_PATH_TRAINING"],
        experiment_name=config["MLFLOW_CONFIGS"]["EXPERIMENT_NAME_TRAINING"],
    )
    mlflow_obj.close()
    mlflow_obj.start()  # start the session

    run_id = str(
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .tags()
        .get("jobRunId")
        .get()
    )

    """ DQM Checks"""
    dqm_checks(config, catalog_name, main_logger)

    """ Meta Data Load """
    main_logger.info("Beginning to load the data..")
    data_dict_read = read_data(config, catalog_name, main_logger)
    main_logger.info("Loaded the data successfully!")
    main_logger.info("===========================================")

    """ Data Load & processing """
    config_obj = ConfigDetails(data_dict_read)
    """ filter the master data and asp data to latest 23 quarters
    and get the forecast_start_month based on max month of asp actuals"""
    # read the master_data and asp_data input tables
    master_data = config_obj.master_data
    asp_data = config_obj.asp_data
    # convert the cal_dt to datetime format
    master_data["CAL_DT"] = pd.to_datetime(master_data["CAL_DT"], errors="coerce")
    # Filter the master data with latest 23 quarters
    # 15 quarters are considered for training and latest
    # 8 quarters are considered for test/validation.
    max_date = master_data["CAL_DT"].max()
    start_date = max_date - pd.offsets.QuarterBegin(23)
    training_month_start = start_date + pd.DateOffset(months=1)
    master_data = master_data[(master_data["CAL_DT"] >= training_month_start)]
    asp_data = asp_data[(asp_data["CAL_DT"] >= training_month_start)]
    training_month_end = max_date - pd.offsets.QuarterEnd(8)
    forecast_month_start = training_month_end + pd.DateOffset(months=1)
    forecast_month_start = forecast_month_start.strftime("%Y-%m")
    main_logger.info(f"forecast month:{forecast_month_start}")

    """ Detect Outliers from Master Data and Impute the Outlier values"""
    main_logger.info("Outlier Detection for Master_data started")
    master_data_imputed = outlier_detection(
        master_data=master_data,
        lst_prioritized_products=config_obj.lst_prioritized_jcodes,
        feature_name=config["OUTLIER_MODULE"]["FEATURE_INFO_1"]["FEATURE_NAME"],
        data_freq=config["OUTLIER_MODULE"]["FEATURE_INFO_1"]["DATA_FREQ"],
        impute_method=config["OUTLIER_MODULE"]["FEATURE_INFO_1"]["IMPUTE_METHOD"],
        logger=main_logger,
    )
    main_logger.info("Outlier Detection and Imputation completed successfully!")

    main_logger.info("Processing Master, EP, ASP dataframes")
    # process the dataframe
    (
        master_data_processed,
        ep_data_processed,
        market_event_data_processed,
        actual_asp_monthly,
        actual_asp_quarterly,
    ) = data_process(
        config,
        config_obj,
        main_logger,
        master_data_imputed,
        asp_data,
        forecast_month_start,
    )
    post_process_path = config["DATA"]["files_location"] + constants.POSTPROCESS_FOLDER

    main_logger.info("Writing Processed data to the folder.")
    actual_asp_quarterly.to_csv(
        post_process_path + "actual_asp_quarterly.csv", sep=",", index=False
    )
    actual_asp_monthly.to_csv(
        post_process_path + "actual_asp_monthly.csv", sep=",", index=False
    )
    market_event_data_processed.to_csv(
        post_process_path + "market_event_data_processed.csv", sep=",", index=False
    )

    # Get configuration dictionaries
    main_logger.info("Getting Configuration...")
    config_dict_generic = get_config(
        config=config,
        generic_flag=True,
        forecast_months=config_obj.forecast_months,
    )
    config_dict_non_generic = get_config(
        config=config,
        generic_flag=False,
        forecast_months=config_obj.forecast_months,
    )
    # tag normal configuration to configuration dictionary as well
    main_logger.info("Updating configuration dictionary")
    config_dict_generic.update(config)
    config_dict_non_generic.update(config)
    main_logger.debug(
        "Configuration set for Generic Product:{}".format(config_dict_generic)
    )
    main_logger.debug(
        "Configuration set for Non-Generic Product:{}".format(config_dict_non_generic)
    )

    """ Getting the train and test date range information to process the data"""
    (
        train_months,
        test_months,
        train_quarters,
        test_quarters,
    ) = create_train_test_month_quarters(
        start_forecast_month=forecast_month_start,
        forecast_months=config_obj.forecast_months,
        processed_master_data=master_data_processed,
        logger=main_logger,
    )
    forecast_month_end = pd.to_datetime(forecast_month_start) + pd.DateOffset(
        months=config_obj.forecast_months - 1
    )

    # Process generic products
    master_data_processed_generic = master_data_processed[
        master_data_processed["J_CODE"].isin(
            master_data_processed[master_data_processed["PROD_SGMNT"] == "GENERICS"][
                "J_CODE"
            ].unique()
        )
    ]
    master_data_processed_generic = master_data_processed_generic[
        master_data_processed_generic["ASP_MTH"] <= forecast_month_end
    ]

    """
    Initialize an empty list to hold the model records as dictionaries
    It will hold the details of models like model_name, model version number,
    baseline_dataset_path that was used to train the model.
    """
    model_records = []

    """
    Initialize an empty list to hold the details of model for registring to mlflow
    It will hold all the details like model_name, generic_flag, scaler object, 
    model signature, xgb parameters.
    """
    registry_records = []

    # As this script is used for re-training the model,
    # we are setting the operation mode as 'retraining'.
    # Train and predict the ASP using ARIMA & XGB models for generics products.
    """ Processing """
    main_logger.info("Training the models and predicting the forecast...")
    main_logger.info("Forecast for GENERIC products...")
    predictions_generic, champion_version = forecast_asp(
        master_data_processed=master_data_processed_generic,
        market_event_data_processed=market_event_data_processed,
        operation_mode="retraining",
        model_alias=None,
        config=config,
        config_dict=config_dict_generic,
        test_months=test_months,
        train_quarters=train_quarters,
        generic_flag=True,
        mlflow_obj=mlflow_obj,
        categorical_cols=config["MODEL_FEATURES"]["GENERIC_PRODUCT"][
            "CATEGORICAL_COLUMNS"
        ],
        model_records=model_records,
        registry_records=registry_records,
        data_drift_records=None,
        catalog_name=catalog_name,
        run_id=run_id,
        logger=main_logger,
    )

    main_logger.info("Forecast for generic products is done successfully!")
    main_logger.info("===========================================")

    """ Forecast - Non Generic Product """
    main_logger.info("Forecast for Non-generic products begins...")

    if config_dict_non_generic["forecast_for_priotized_products_only"]:
        master_data_processed_non_generic = master_data_processed[
            master_data_processed["J_CODE"].isin(config_obj.lst_prioritized_jcodes)
        ]
    else:
        master_data_processed_non_generic = master_data_processed

    master_data_processed_non_generic = master_data_processed_non_generic[
        master_data_processed_non_generic["ASP_MTH"] <= forecast_month_end
    ]

    # Train and predict the ASP using ARIMA & XGB models for generics products.
    predictions_non_generic, champion_version = forecast_asp(
        master_data_processed=master_data_processed_non_generic,
        market_event_data_processed=market_event_data_processed,
        operation_mode="retraining",
        model_alias=None,
        config=config,
        config_dict=config_dict_non_generic,
        test_months=test_months,
        train_quarters=train_quarters,
        generic_flag=False,
        mlflow_obj=mlflow_obj,
        categorical_cols=config["MODEL_FEATURES"]["NON_GENERIC_PRODUCT"][
            "CATEGORICAL_COLUMNS"
        ],
        model_records=model_records,
        registry_records=registry_records,
        data_drift_records=None,
        catalog_name=catalog_name,
        run_id=run_id,
        logger=main_logger,
    )
    main_logger.info("Forecast for Non-generic products is done successfully!")

    # Filter and combine predictions
    if config_dict_non_generic["forecast_for_priotized_products_only"]:
        predictions_generic = predictions_generic[
            predictions_generic["J_CODE"].isin(config_obj.lst_prioritized_jcodes)
        ]

    predictions_non_generic = predictions_non_generic[
        ~predictions_non_generic["J_CODE"].isin(predictions_generic["J_CODE"].unique())
    ]
    predictions_all = pd.concat([predictions_generic, predictions_non_generic], axis=0)
    test_quarters_to_report = list(set(test_quarters) - set(train_quarters))

    predictions_all["QUARTER"] = predictions_all["QUARTER"].astype(str)
    predictions_all = predictions_all[
        predictions_all["QUARTER"].isin(test_quarters_to_report)
    ]
    # Save predictions dataframe
    main_logger.info("Writing Prediction results to the folder")
    predictions_path = config["DATA"]["files_location"] + constants.RESULTS_FOLDER
    predictions_generic.to_csv(
        predictions_path + "predictions_generic.csv", sep=",", index=False
    )
    predictions_non_generic.to_csv(
        predictions_path + "predictions_non_generic.csv", sep=",", index=False
    )
    predictions_all.to_csv(predictions_path + "predictions.csv", sep=",", index=False)

    """ Post-processing """
    main_logger.info("Post processing the predictions...")
    market_event_data_processed["ASP_MTH"] = (
        pd.to_datetime(market_event_data_processed["ASP_MTH"])
        .dt.strftime("%m/%d/%Y")
        .astype(str)
    )
    market_event_data_processed_generic = market_event_data_processed[
        market_event_data_processed["J_CODE"].isin(
            master_data_processed_generic["J_CODE"].unique()
        )
    ]
    market_event_data_processed_generic.to_csv(
        post_process_path + "market_event_data_processed_generic.csv",
        sep=",",
        index=False,
    )
    POST_PROCESS_FOLDER_PATH = (
        config["DATA"]["files_location"] + constants.POSTPROCESS_FOLDER
    )
    RESULT_FOLDER_PATH = config["DATA"]["files_location"] + constants.RESULTS_FOLDER
    predictions_path = config["DATA"]["files_location"] + constants.RESULTS_FOLDER

    actual_asp_quarterly = pd.read_csv(
        POST_PROCESS_FOLDER_PATH + "actual_asp_quarterly.csv"
    )
    predictions_all = pd.read_csv(RESULT_FOLDER_PATH + "predictions.csv")
    market_event_data_processed_generic = pd.read_csv(
        POST_PROCESS_FOLDER_PATH + "market_event_data_processed_generic.csv"
    )

    predictions_post_processed = post_process(
        df_actual_train_val=actual_asp_quarterly,
        df_model_forecast=predictions_all,
        df_market_event=market_event_data_processed_generic,
        logger=main_logger,
    )

    # Rename the column ASP_PRED_SCALED to ASP_PRED
    predictions_post_processed = predictions_post_processed.rename(
        columns={"ASP_PRED_SCALED": "ASP_PRED"}
    )
    
    predictions_post_processed.to_csv(
        predictions_path + "final_predictions.csv", sep=",", index=False
    )
    main_logger.info("Post processing successfully done!")

    predictions_post_processed["QUARTER_START_DATE"] = (
        predictions_post_processed["QUARTER"]
        .apply(lambda x: pd.Period(x, freq="Q").start_time)
        .dt.date
    )
    predictions_post_processed["QUARTER_END_DATE"] = (
        predictions_post_processed["QUARTER"]
        .apply(lambda x: pd.Period(x, freq="Q").end_time)
        .dt.date
    )

    main_logger.info("Registering the models to model registry")

    # Register all the models (generic-scaled,nongeneric-scaled) in model registry.
    # All these models are created in the unity catalog
    # save the trained model, and objects together to track in mlflow
    main_logger.info("Creating Custom BaseModel to Log into MLflow")
    for record in registry_records:
        trained_model = record["trained_model"]
        generic_flag = record["generic_flag"]
        scaler_obj = record["scaler"]
        sign = record["model_signature"]
        default_xgb_params = record["xgb_params"]
        save_model_mlflow(
            trained_model,
            generic_flag,
            scaler_obj,
            mlflow_obj,
            config,
            sign,
            run_id,
            default_xgb_params,
            model_records,
            catalog_name,
            main_logger,
        )

    merged_df = pd.merge(
        actual_asp_quarterly,
        predictions_post_processed,
        on=["J_CODE", "QUARTER"],
        how="inner",
    )

    # Drop rows with NaN values in 'ASP_TRUE' or 'ASP_PRED'
    merged_df = merged_df.dropna(subset=["ASP_TRUE", "ASP_PRED"])

    # Calculate overall RMSE of all the validation quarters
    overall_rmse = np.sqrt(
        mean_squared_error(merged_df["ASP_TRUE"], merged_df["ASP_PRED"])
    )

    # Calculate overall RMSE of all the validation quarters
    overall_mape = mean_absolute_percentage_error(
        merged_df["ASP_TRUE"], merged_df["ASP_PRED"]
    )

    # Calculate quarterly RMSE
    quarterly_rmse = (
        merged_df.groupby("QUARTER", group_keys=False)
        .apply(lambda x: np.sqrt(mean_squared_error(x["ASP_TRUE"], x["ASP_PRED"])))
        .to_dict()
    )

    # Calculate quarterly MAPE
    quarterly_mape = (
        merged_df.groupby("QUARTER", group_keys=False)
        .apply(lambda x: mean_absolute_percentage_error(x["ASP_TRUE"], x["ASP_PRED"]))
        .to_dict()
    )

    # Prepare the metrics dictionary
    metrics = {
        "Overall_RMSE": overall_rmse,
        "Overall_MAPE": overall_mape,
        **{f"RMSE_Quarter_{k}": v for k, v in quarterly_rmse.items()},
        **{f"MAPE_Quarter_{k}": v for k, v in quarterly_mape.items()},
    }

    """ Track all these into mlflow parameters """
    mlflow_obj.log_artifacts(
        data_dict_read["files_location"] + constants.MODELS_FOLDER, "model_results"
    )
    mlflow_obj.log_artifacts(
        data_dict_read["files_location"] + constants.RESULTS_FOLDER, "results"
    )

    mlflow_obj.log_artifact(
        data_dict_read["files_location"] + constants.LOGS_FOLDER + log_file, "logs"
    )

    mlflow_obj.log_artifacts(
        data_dict_read["files_location"] + constants.POSTPROCESS_FOLDER,
        "postprocess_files",
    )

    mlflow_obj.log_mlflow_metrics(metrics)

    # Define schema with the desired column order
    # to convert the dataframe to a Delta table
    baseline_schema = StructType(
        [
            StructField("run_id", StringType(), True),
            StructField("model_name", StringType(), True),
            StructField("version_number", StringType(), True),
            StructField("baseline_dataset_path", StringType(), True),
        ]
    )
    baseline_dataset_df = spark.createDataFrame(model_records, schema=baseline_schema)

    # create the delta table that stores details of baseline datasets for all the models
    baseline_table_name = config["OUTPUT"]["BASELINE_TABLE"]
    baseline_table_name = f"{catalog_name}.edp_psas_di_usp_gold.{baseline_table_name}"
    baseline_delta_table_path = (
        f"abfss://usp@{storage_account}.dfs.core.windows.net/"
        f"gold/edp_psas_di_usp_gold/ASP_FORECAST/{baseline_table_name}"
    )

    # Check if Delta table exists by trying to read the path
    try:
        spark.sql(f"DESCRIBE TABLE {baseline_table_name}")
        main_logger.info(
            f"Delta table {baseline_table_name} "
            f"exists in Unity Catalog, appending data."
            )

        # Append new records to the Delta table
        baseline_dataset_df.write.format("delta").mode("append").option(
            "path", baseline_delta_table_path
        ).saveAsTable(baseline_table_name)

    except Exception:
        main_logger.info(
            f"Delta table {baseline_table_name} does not exist in Unity Catalog, "
            f"creating new Delta table."
        )

        # Create new Delta table and write the data
        baseline_dataset_df.write.format("delta").mode("overwrite").option(
            "path", baseline_delta_table_path
        ).saveAsTable(baseline_table_name)

    # get the latest model version number from model_records list.
    model_version_number = 999999
    for record in model_records:
        if model_version_number > int(record["version_number"]):
            model_version_number = int(record["version_number"])

    # Create the final df by combining the master data and forecast
    # predictions from the model for creating model_validation table.
    master_data = config_obj.master_data
    master_data["QUARTER"] = (
        pd.to_datetime(master_data["CAL_DT"]).dt.to_period("Q").astype("str")
    )
    master_data = spark.createDataFrame(master_data)
    asp_master_data_transformed = master_data.groupBy(
        "J_CODE", "QUARTER", round("ASP_PRC", 3).alias("ASP_PRC")
    ).agg(
        upper(min_("PROD_NAME")).alias("PROD_NAME"),
        upper(min_("PROD_SGMNT")).alias("PROD_SGMNT"),
        upper(min_("THERA_CLS_DSCR")).alias("THERA_CLS_DSCR"),
        upper(min_("DISEASE_ST")).alias("DISEASE_ST"),
    )

    merged_df_spark = spark.createDataFrame(merged_df)
    merged_df_spark = (
        merged_df_spark.withColumn("MODEL_TRAINED_FROM", lit(training_month_start))
        .withColumn("MODEL_TRAINED_TILL", lit(training_month_end))
        .withColumn("MODEL_TRAINED_ON", lit(datetime.now()))
        .withColumn("MODEL_VERSION_NUMBER", lit(model_version_number))
        .withColumn("RUN_ID", lit(run_id))
    )

    # Perform the join between `merged_df` and the transformed `asp_master_data`
    result_df = merged_df_spark.alias("m").join(
        asp_master_data_transformed.alias("asp"), on=["J_CODE", "QUARTER"], how="left"
    )

    # Select the required columns and alias ASP_TRUE and ASP_PRED
    final_df = result_df.select(
        "m.RUN_ID",
        "m.MODEL_VERSION_NUMBER",
        "m.J_CODE",
        "asp.PROD_NAME",
        "asp.PROD_SGMNT",
        col("m.QUARTER").alias("QUARTER"),
        col("m.ASP_PRED").alias("ASP_PRC_FORECASTED"),
        col("asp.ASP_PRC").alias("ASP_PRC_ACTL"),
        "m.QUARTER_START_DATE",
        "m.QUARTER_END_DATE",
        "m.MODEL_TRAINED_FROM",
        "m.MODEL_TRAINED_TILL",
        "m.MODEL_TRAINED_ON",
        "asp.THERA_CLS_DSCR",
        "asp.DISEASE_ST",
    )

    validation_table_name = config["OUTPUT"]["VALIDATION_TABLE"]
    full_table_name = f"{catalog_name}.edp_psas_di_usp_gold.{validation_table_name}"
    delta_table_path = (
        f"abfss://usp@{storage_account}.dfs.core.windows.net/gold/"
        f"edp_psas_di_usp_gold/ASP_FORECAST/{validation_table_name}"
    )
    # Check if Delta table exists by trying to read the path
    try:
        spark.sql(f"DESCRIBE TABLE {full_table_name}")
        main_logger.info(
            f"Delta table {full_table_name} "
            f"exists in Unity Catalog, appending data."
            )

        # Append new records to the Delta table
        final_df.write.format("delta").mode("append").option(
            "path", delta_table_path
        ).saveAsTable(full_table_name)

    except Exception:
        main_logger.info(
            f"Delta table {full_table_name} does not exist in Unity Catalog, "
            f"creating new Delta table."
        )

        # Create new Delta table and write the data
        final_df.write.format("delta").mode("overwrite").option(
            "path", delta_table_path
        ).saveAsTable(full_table_name)


if __name__ == "__main__":
    main()
