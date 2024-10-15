import pandas as pd
import json
import os
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from utils import constants
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestNumberOfDriftedColumns,
)
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("asp-forecasting").getOrCreate()


def calculate_drift(
    current_baseline_df: pd.DataFrame,
    champion_baseline_path: str,
    root_folder: str,
    logger: logging.Logger,
):
    """
    Calculate data drift and save reports.

    Args:
        current_baseline_df (pd.DataFrame): DataFrame containing the current baseline data.
        champion_baseline_path (str): Path to the file containing the champion baseline data.
        root_folder (str): Root directory for saving results.
        logger (logging.Logger): Logger for recording the process.
    Returns:
        None
    """
    child_logger = logger.getChild("GENERATING DATA DRIFT REPORTS")

    # Load the champion baseline data
    champion_baseline_df = spark.read.csv(
        champion_baseline_path, header=True, inferSchema=True
    ).toPandas()
    champion_baseline_df = champion_baseline_df.iloc[:, 2:]

    # Ensure both dataframes have the same structure by selecting matching columns
    current_baseline_df = current_baseline_df[champion_baseline_df.columns]

    # Align data types between the two dataframes
    dtype_mapping = champion_baseline_df.dtypes.to_dict()
    current_baseline_df = current_baseline_df.astype(dtype_mapping)
    current_baseline_df.reset_index(drop=True, inplace=True)
    champion_baseline_df.reset_index(drop=True, inplace=True)
    # Log the shapes and column information
    child_logger.info(
        f"Champion Baseline dataframe shape: {champion_baseline_df.shape}"
    )
    child_logger.info(f"Current Dataframe shape: {current_baseline_df.shape}")

    # Generate and save the data drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=champion_baseline_df, current_data=current_baseline_df)
    drift_as_dict = report.as_dict()

    # Define the path for the folder where the drift report will be saved
    drift_folder_path = f"{root_folder}{constants.DATA_DRIFT_FOLDER}"

    # Define the path for saving the JSON version of the data drift report
    json_file_path = os.path.join(drift_folder_path, "data_drift_report.json")

    # Ensure the drift folder exists, creating it if necessary
    os.makedirs(drift_folder_path, exist_ok=True)

    # Save the drift report as a JSON file
    with open(json_file_path, "w", encoding="utf8") as fp:
        json.dump(drift_as_dict, fp, indent=2, ensure_ascii=False)

    # Save the HTML version of the drift report for easy viewing
    report.save_html(os.path.join(drift_folder_path, "data_drift_report.html"))

    # Run data stability tests and save the results
    child_logger.info("Running Data Stability Tests...")
    tests = TestSuite(
        tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ]
    )

    # Run stability tests comparing champion baseline data with the current baseline data
    tests.run(reference_data=champion_baseline_df, current_data=current_baseline_df)
    # Save the results of the stability tests as an HTML file for easy viewing
    tests.save_html(os.path.join(drift_folder_path, "data_stability_tests.html"))
