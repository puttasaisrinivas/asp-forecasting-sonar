import pandas as pd
from utils import constants
from utils.helpers import get_config_from_file, read_data, get_config
from utils.logger import setup_logger
from utils.dir import cleanup_dir, setup_dir
from utils.forecast_validation import forecast_validation
from extract import data_process
from config import ConfigDetails
from processing.process import create_train_test_month_quarters
from processing.post_processing import post_process
from train import forecast_asp
from utils.track import MLFlowManager
import sys
from utils.dqm import dqm_checks
from utils.outlier_detection import outlier_detection
import mlflow
from datetime import datetime
from pyspark.sql import Window
from pyspark.sql.functions import col, lit, row_number, min as min_, upper
from utils.data_drift import calculate_drift
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local").appName("asp-forecasting").getOrCreate()


def inference_main():
    """
    Main function to load a logged MLflow model and make predictions on test data.

    """

    """Load configuration File"""
    config = get_config_from_file(constants.CONFIG_FOLDER_PATH + "general.yaml")

    """Setup and clean dir"""
    cleanup_dir(config)
    setup_dir(config)

    params = sys.argv[1:]  # Exclude the script name

    # Parse parameters directly
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
    mlflow.set_registry_uri("databricks-uc")
    mlflow_obj = MLFlowManager(
        experiment_path=config["MLFLOW_CONFIGS"]["EXPERIMENT_PATH_INFERENCE"],
        experiment_name=config["MLFLOW_CONFIGS"]["EXPERIMENT_NAME_INFERENCE"],
    )

    run_id = str(
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .tags()
        .get("jobRunId")
        .get()
    )
    mlflow_obj.close()
    mlflow_obj.start()  # start the session

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
    # get the forecast_start_month based on max month of asp actuals
    # read the master and asp data
    master_data = config_obj.master_data
    asp_data = config_obj.asp_data
    master_data["CAL_DT"] = pd.to_datetime(master_data["CAL_DT"], errors="coerce")
    # get max_date from master data for filtering the last 23 quarters.
    max_date = master_data["CAL_DT"].max()
    start_date = max_date - pd.offsets.QuarterBegin(23)
    master_data = master_data[(master_data["CAL_DT"] > start_date)]
    asp_data = asp_data[(asp_data["CAL_DT"] > start_date)]
    # forecast start month is first month of future data where we don't have actual ASP.
    forecast_month_start = max_date + pd.DateOffset(months=1)
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
    initializing the list to store baseline dataset details 
    of each model as a dictionary. It will hold details such as 
    current baseline_dataset_path and champion baseline_dataset_path for calculating
    the drift in the data if any, root folder to store the reports.
    """
    data_drift_records = []

    # As this script is used for inference ,
    # we are setting the operation mode as 'inference'.
    # Train and predict the ASP using ARIMA & XGB models for generics products.
    """ Processing """
    main_logger.info("Training the models and predicting the forecast...")
    main_logger.info("Forecast for GENERIC products...")
    predictions_generic, model_version = forecast_asp(
        master_data_processed=master_data_processed_generic,
        market_event_data_processed=market_event_data_processed,
        operation_mode="inference",
        model_alias ="champion",
        config=config,
        config_dict=config_dict_generic,
        test_months=test_months,
        train_quarters=train_quarters,
        generic_flag=True,
        mlflow_obj=mlflow_obj,
        categorical_cols=config["MODEL_FEATURES"]["GENERIC_PRODUCT"][
            "CATEGORICAL_COLUMNS"
        ],
        model_records=None,
        registry_records=None,
        data_drift_records=data_drift_records,
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
    # As this script is used for inference ,
    # we are setting the operation mode as 'inference'.
    # Train and predict the ASP using ARIMA & XGB models for non-generic products.
    predictions_non_generic, model_version = forecast_asp(
        master_data_processed=master_data_processed_non_generic,
        market_event_data_processed=market_event_data_processed,
        operation_mode="inference",
        model_alias="champion",
        config=config,
        config_dict=config_dict_non_generic,
        test_months=test_months,
        train_quarters=train_quarters,
        generic_flag=False,
        mlflow_obj=mlflow_obj,
        categorical_cols=config["MODEL_FEATURES"]["NON_GENERIC_PRODUCT"][
            "CATEGORICAL_COLUMNS"
        ],
        model_records=None,
        registry_records=None,
        data_drift_records=data_drift_records,
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

    predictions_post_processed = spark.createDataFrame(predictions_post_processed)

    main_logger.info(f"Total Number of data drift records: {len(data_drift_records)}")

    # calculate data drift for all the models
    for record in data_drift_records:
        current_baseline = record["current_baseline_path"]
        champion_baseline = record["champion_baseline_path"]
        root_folder = record["root_folder"]

        # Call the drift calculation function
        calculate_drift(current_baseline, champion_baseline, root_folder, main_logger)

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

    # Calculate the forecast confidence score for all the products
    validation_table = config["OUTPUT"]["VALIDATION_TABLE"]
    model_validation_df = spark.read.table(
        f"{catalog_name}.edp_psas_di_usp_gold.{validation_table}"
    )
    model_trained_on = (
        model_validation_df.filter(col("MODEL_VERSION_NUMBER") == model_version)
        .select("MODEL_TRAINED_ON")
        .first()["MODEL_TRAINED_ON"]
    )
    model_trained_on = str(pd.to_datetime(model_trained_on).to_period("Q"))
    confidence_df = forecast_validation(
        model_validation_df, market_event_data_processed, model_version, main_logger
    )

    master_data = config_obj.master_data
    master_data = spark.createDataFrame(master_data)
    # Define a window specification to partition by J_CODE
    # and order by Quarter descending
    window_spec = Window.partitionBy("J_CODE").orderBy(col("CAL_DT").desc())

    # Add a row number to each row within the partition
    df_with_row_number = master_data.withColumn(
        "row_number", row_number().over(window_spec)
    )

    # Filter to get only the rows where row_number is 1 (latest Quarter for each J_CODE)
    latest_asp_price_df = df_with_row_number.filter(col("row_number") == 1).select(
        "J_CODE", "ASP_PRC"
    )
    # master_data = config_d
    asp_master_data_transformed = master_data.groupBy("J_CODE").agg(
        upper(min_("PROD_NAME")).alias("PROD_NAME"),
        upper(min_("PROD_SGMNT")).alias("PROD_SGMNT"),
        upper(min_("THERA_CLS_DSCR")).alias("THERA_CLS_DSCR"),
        upper(min_("DISEASE_ST")).alias("DISEASE_ST"),
    )
    asp_master_data_transformed = asp_master_data_transformed.join(
        latest_asp_price_df, on="J_CODE", how="inner"
    )

    asp_master_data_transformed = (
        asp_master_data_transformed.withColumn("RUN_ID", lit(run_id))
        .withColumn("INFERENCE_DATE", lit(datetime.now()))
        .withColumn("MODEL_VERSION", lit(model_version))
        .withColumn("MODEL_TRAINED_ON", lit(model_trained_on))
    )

    # Perform the join between `merged_df` and the final forecast df
    result_df = (
        predictions_post_processed.alias("asp")
        .join(
            asp_master_data_transformed.alias("m"),
            col("m.J_CODE") == col("asp.J_CODE"),
            how="left",
        )
        .join(
            confidence_df.alias("cnfdnce"),
            col("asp.J_CODE") == col("cnfdnce.J_CODE"),
            how="left",
        )
    )

    result_df = result_df.withColumn(
        "PRICE_CHANGE_WITH_CP",
        (col("asp.ASP_PRED") - col("m.ASP_PRC")) / col("m.ASP_PRC") * 100,
    )

    final_df = result_df.select(
        "m.RUN_ID",
        "asp.J_CODE",
        "m.PROD_NAME",
        "m.PROD_SGMNT",
        col("asp.QUARTER").alias("QUARTER"),
        col("asp.ASP_PRED").alias("ASP_PRC_FORECASTED"),
        col("m.ASP_PRC").alias("CURRENT_ASP_PRC"),
        "PRICE_CHANGE_WITH_CP",
        "asp.QUARTER_START_DATE",
        "asp.QUARTER_END_DATE",
        "m.THERA_CLS_DSCR",
        "m.DISEASE_ST",
        "cnfdnce.CONFIDENCE_SCORE",
        "cnfdnce.CONFIDENCE_CATEGORY",
        "m.INFERENCE_DATE",
        "m.MODEL_VERSION",
        "m.MODEL_TRAINED_ON",
    )

    # create a delta table for storing the predictions from inference-champion pipeline
    # if table already exists, move the existing records to historic table.
    champion_table_current = config["OUTPUT"]["CHAMPION_TABLE"]
    full_table_name_current = (
        f"{catalog_name}.edp_psas_di_usp_gold.{champion_table_current}"
    )
    delta_table_path_current = (
        f"abfss://usp@{storage_account}.dfs.core.windows.net/gold/"
        f"edp_psas_di_usp_gold/ASP_FORECAST/{champion_table_current}"
    )

    # get the parameters for champion historic table
    champion_table_hist = champion_table_current + "_HIST"
    full_table_name_hist = f"{catalog_name}.edp_psas_di_usp_gold.{champion_table_hist}"
    delta_table_path_hist = (
        f"abfss://usp@{storage_account}.dfs.core.windows.net/gold/"
        f"edp_psas_di_usp_gold/ASP_FORECAST/{champion_table_hist}"
    )

    # Check if Delta table exists by trying to read the path
    try:
        spark.sql(f"DESCRIBE TABLE {full_table_name_current}")
        main_logger.info(
            f"Delta table {full_table_name_current} exists in Unity Catalog, "
            f"moving the existing data to historic table and "
            f"overwriting the table with records of current run."
        )

        previous_df = spark.sql(f"SELECT * FROM {full_table_name_current}")

        try:
            spark.sql(f"DESCRIBE TABLE {full_table_name_hist}")
            main_logger.info(
                f"Historic Delta table {full_table_name_hist} exists in Unity Catalog, "
                f"appending the existing records to the historic table."
            )

            # Append new records to the Delta table
            previous_df.write.format("delta").mode("append").option(
                "path", delta_table_path_hist
            ).saveAsTable(full_table_name_hist)

        except Exception:
            main_logger.info(
                f"Historic Delta table {full_table_name_hist} "
                f"does not exist in Unity Catalog, creating new Delta table."
            )

            # Create new Delta table and write the data
            previous_df.write.format("delta").mode("overwrite").option(
                "path", delta_table_path_hist
            ).saveAsTable(full_table_name_hist)

        # Append new records to the Delta table
        final_df.write.format("delta").mode("overwrite").option(
            "path", delta_table_path_current
        ).saveAsTable(full_table_name_current)

    except Exception:
        main_logger.info(
            f"Delta table {full_table_name_current} does not exist in Unity Catalog, "
            f"creating new Delta table."
        )

        # Create new Delta table and write the data
        final_df.write.format("delta").mode("overwrite").option(
            "path", delta_table_path_current
        ).saveAsTable(full_table_name_current)

    return 0


inference_main()
