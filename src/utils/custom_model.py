import logging
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import mlflow
import yaml
import mlflow.pyfunc
import pickle
from utils import constants
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder.appName("asp-forecasting").getOrCreate()

# MLFLOW custom model to log scaler + xgboost as single artifact
class ScalerXGBModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        scaler_path = str(Path(context.artifacts["scaler"] + "/model.pkl"))
        self.scaler = pickle.load(open(scaler_path, "rb"))
        xgb_model_path = str(Path(context.artifacts["xgb_model"] + "/model.xgb"))
        self.xgb_model = XGBRegressor()  # Initialize the XGBRegressor object
        self.xgb_model.load_model(xgb_model_path)  # Load the model from the .xgb file

    def predict(self, context, model_input):
        # Ensure model_input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        return self.xgb_model.predict(model_input)


def save_model_mlflow(
    trained_model: XGBRegressor,
    generic_flag: bool,
    scaler_obj: StandardScaler,
    mlflow_obj,
    config: yaml,
    sign,
    run_id: str,
    default_xgb_params: dict,
    model_records: list,
    catalog_name: str,
    logger: logging.Logger,
):
    """
    Creates custom MLFLow Base model to log into MLFlow model registry.
    Our pipelines uses scaled input to predict,
    so saving the model with in-built scaled transformation.

    Args:
    trained_model (XGBRegressor): Trained Model
    generic_flag (bool): flag that gives if data is generic or non-generic.
    scaler_obj (StandardScaler): Scaler object if data is scaled.
    mlflow_obj: MLflow object for logging.
    config (yaml): A Config object initialized with the data from the YAML file.
    sign (ModelSignature): Model signature.
    run_id (str): Pipeline run-id.
    default_xgb_params (dict): Default XGBoost model parameters.
    model_records (list): list to hold model records as dictionaries.
    catalog_name (str): name of the catalog.
    logger (logging.Logger): Logger object.
    """

    child_logger = logger.getChild("CREATE_CUSTOM_MODEL")

    # create custom model
    child_logger.info("Artifact URI:{}".format(mlflow_obj.artifact_uri))
    # Log the model with artifacts
    if generic_flag:
        model_name = "generic-scaled"
    else:
        model_name = "nongeneric-scaled"


    # Set registry URI to Unity Catalog
    mlflow_obj.set_registry_uri("databricks-uc")
    schema = "edp_psas_di_usp_gold"
    full_model_name = f"{catalog_name}.{schema}.{model_name}"

    mlflow_obj.log_mlflow_params(default_xgb_params)
    mlflow_obj.add_tags(run_id)
    # Log the model with a custom ScalerXGBModel if a scaler is provided
    if scaler_obj:
        root_path = config["DATA"]["files_location"] + constants.MODELS_FOLDER
        if generic_flag:
            model_path = root_path + "scaled/generic/"
        else:
            model_path = root_path + "scaled/non-generic/"
        # Create an instance of ScalerXGBModel
        scaler_xgb_model = ScalerXGBModel()
        mlflow_obj.save_and_log_model(
            scaler_obj,
            trained_model,
            scaler_xgb_model,
            sign,
            full_model_name,
            model_path,
        )
    else:
        # Log the XGBoost model directly if no scaler is provided
        mlflow_obj.log_xgb_model(trained_model, full_model_name, sign)

    # Register the model in the MLflow Model Registry
    artifact_uri = mlflow_obj.get_artifact_uri(full_model_name)
    child_logger.info(f"Registering the Model to registry - {full_model_name}")
    child_logger.info(f"artifact URI:{artifact_uri}")
    mlflow_obj.register_model_and_set_champion(artifact_uri, full_model_name, model_records, run_id)


def load_model_for_inference(
    generic_flag: bool,
    mlflow_obj,
    config: yaml,
    catalog_name: str,
    model_alias: str,
    logger: logging.Logger,
):
    """
    Load the model for inference based on the model name.

    Args:
        generic_flag (bool): flag that gives if data is generic or non-generic.
        config (yaml): A Config object initialized with the data from the YAML file
        catalog_name (str): Name of the unity catalog.
        model_alias (str): Alias of the model in the MLflow registry to be loaded.
        mlflow_obj : mlflow_obj to load the trained model from registry.

    Returns:
        Loaded model object for inference.
    """
    child_logger = logger.getChild("LOAD_MODEL_FOR_INFERENCE")

    # Log the model with artifacts
    if generic_flag:
        model_name = "generic-scaled"
    else:
        model_name = "nongeneric-scaled"

    schema = "edp_psas_di_usp_gold"
    full_model_name = f"{catalog_name}.{schema}.{model_name}"

    if model_alias == "champion":
        # to load model using model alias
        model_uri = f"models:/{full_model_name}@{model_alias}"
    else:
        # to load model using model version
        model_uri = f"models:/{full_model_name}/{model_alias}"

    # Load the model
    loaded_model = mlflow_obj.load_pyfunc_model(model_uri)

    # Get the input signature
    input_signature = loaded_model.metadata.get_input_schema()

    # Extract feature names from the input signature
    feature_names = [input.name for input in input_signature.inputs]
    feature_dtypes = {input.name: input.type for input in input_signature.inputs}
    child_logger.info(f"Loaded the model with URI:{model_uri}")

    if model_alias == "champion":
        model_version = mlflow_obj.get_model_version_by_alias(
            full_model_name, model_alias
        ).version
        # get the champion baseline dataset path from lookup table
        baseline_table = config["OUTPUT"]["BASELINE_TABLE"]
        baseline_df = spark.read.table(f"{catalog_name}.{schema}.{baseline_table}")

        # Filter based on 'model_name' and 'version_number'
        filtered_df = baseline_df.filter(
            (col("model_name") == full_model_name)
            & (col("version_number") == model_version)
        )

        # Extract the value of 'baseline_path' as a variable
        champion_baseline_dataset_path = filtered_df.select(
            "baseline_dataset_path"
        ).first()["baseline_dataset_path"]

    else:
        champion_baseline_dataset_path = None
        model_version = model_alias

    return (
        loaded_model,
        feature_names,
        feature_dtypes,
        champion_baseline_dataset_path,
        model_version,
    )
