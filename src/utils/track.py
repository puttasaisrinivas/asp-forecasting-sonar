import os
import shutil
import mlflow
from xgboost import XGBRegressor
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
from pathlib import Path
from mlflow import MlflowClient


def get_mlflow_details():
    """
    Returns details of MLflow if available, else returns "None".

    Returns:
        str: Details of MLflow or "None" if unavailable.
    """
    return "None"


class MLFlowManager:
    def __init__(self, experiment_path: str = None, experiment_name: str = None):
        """
        Initializes the MLFlowManager object.

        Args:
            experiment_path (str, optional): Path to the MLflow experiment.
            experiment_name (str, optional): Name of the MLflow experiment.
        """

        # Store the path to the MLflow experiment
        self.experiment_path = experiment_path

        # Store the name of the MLflow experiment
        self.experiment_name = experiment_name

        # Initialize placeholders for run details
        self.run = None
        self.eid = None
        self.artifact_uri = None

    def start(self):
        """
        Starts a new MLflow run session.

        Returns:
            MLFlowManager: The current MLFlowManager instance.
        """

        # Create experiment path
        if self.experiment_name is None and self.experiment_path is None:
            experiment_path = get_mlflow_details()
        elif self.experiment_name is None:
            # If only experiment_name is None, construct the path from experiment_path
            exp_name = get_mlflow_details().split("/")[-1]
            experiment_path = self.experiment_path + "/{}".format(exp_name)
        elif self.experiment_path is None:
            # Raise an error if experiment_path is None but experiment_name is provided
            raise RuntimeError("Mlflow Experiment Path cannot be None...")
        else:
            experiment_path = self.experiment_path

        # Create or retrieve experiment in MLflow
        try:
            print("Creating New Experiment in MLflow")
            mlflow.create_experiment(
                self.experiment_name, artifact_location=experiment_path
            )
            print("Starting the run session")
            self.eid = mlflow.get_experiment_by_name(self.experiment_name)
            self.run = mlflow.start_run(self.eid.experiment_id)
        except Exception:
            print("MLFlow Resource already Exists.")
            self.eid = mlflow.get_experiment_by_name(self.experiment_name)
            self.run = mlflow.start_run(experiment_id=self.eid.experiment_id)
            pass

        self.artifact_uri = mlflow.get_artifact_uri()

        return self

    def log_mlflow_params(self, params: dict):
        """
        Logs parameters to the current MLflow run.

        Args:
            params (dict): Dictionary of parameters to log.

        Returns:
            MLFlowManager: The current MLFlowManager instance.
        """

        if self.run:
            mlflow.log_params(params)
        else:
            raise RuntimeError(
                """No active MLflow run.
                Please start a run using start_run() first."""
            )
        return self

    def log_mlflow_metrics(self, metrics: dict, epoch=None):
        """
        Logs training metrics to the current MLflow run.

        Args:
            metrics (dict): Dictionary of metrics to log.
            epoch (int, optional): Training epoch (optional).

        Returns:
            MLflowRun: The current instance for chaining methods.
        """

        if self.run:
            if epoch is not None:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value, epoch)
            else:
                mlflow.log_metrics(metrics)
        else:
            raise RuntimeError(
                """No active MLflow run.
                Please start a run using start_run() first."""
            )
        return self

    def log_artifacts(self, artifact_path: str, artifact_dir: str = "data/"):
        """
        Logs training artifacts to the current MLflow run.

        Args:
            artifact_path (str): Path to the artifact to log.
            artifact_dir (str, optional): Subdirectory within the
            run for artifacts. Defaults to "data/".

        Returns:
            MLflowRun: The current instance for chaining methods.
        """

        if self.run:
            mlflow.log_artifacts(artifact_path, artifact_dir)
        else:
            raise RuntimeError(
                "No active MLflow run. Please start a run using start_run() first."
            )
        return self

    def log_artifact(self, artifact_file: str, artifact_dir: str = "data/"):
        """
        Logs training artifacts to the current MLflow run.

        Args:
            artifact_file (str): File name to log as artifact
            artifact_dir (str, optional): Subdirectory within the run for artifacts.
            Defaults to "data/".

        Returns:
            MLflowRun: The current instance for chaining methods.
        """

        if self.run:
            mlflow.log_artifact(artifact_file, artifact_dir)
        else:
            raise RuntimeError(
                "No active MLflow run. Please start a run using start_run() first."
            )
        return self

    def log_xgb_model(
        self, trained_model: XGBRegressor, artifact_path: str, signature=None
    ):
        """
        Logs the model using MLflow Model and optionally registers it in the registry.

        Args:
            trained_model (XGBRegressor): trained model to be logged into mlflow.
            artifact_path (str): path to place the artifacts.
            signature: Model Signature
        Returns:
            MLflowRun: The current instance for chaining methods.
        """

        if self.run:
            mlflow.xgboost.log_model(trained_model, artifact_path, signature=signature)
        else:
            raise RuntimeError(
                "No active MLflow run. Please start a run using start_run() first."
            )
        return self

    def register_model_and_set_champion(
        self,
        artifact_path: str,
        model_name: str,
        model_records: list,
        pipeline_run_id: str,
    ) -> bool:
        """
        Registers the model in the MLflow registry.

        Args:
            artifact_path (str): Path to the model artifact.
            model_name (str): Name of the model.
            model_records (list): list to append model records.
            pipeline_run_id (str): pipeline run id.

        Returns:
            bool: True if model registration is successful, False otherwise.
        """

        run_id = self.run.info.run_id
        registered_model_version = mlflow.register_model(
            model_uri=artifact_path, name=model_name
        )

        # Get the version number of the newly registered model
        version_number = registered_model_version.version

        # Create an MLflowClient to interact with the model registry
        client = MlflowClient()

        # Set alias 'champion' to the newly registered model version
        client.set_registered_model_alias(
            name = model_name,
            alias = "champion",
            version = version_number,
        )

        # split model_name
        model_name_split = model_name.split(".")[2].split("-")
        scaling_flag = model_name_split[1]
        generic_flag = model_name_split[0]

        if generic_flag == "nongeneric":
            generic_flag = "non-generic"

        baseline_dataset_path = (
            f"dbfs:/FileStore/asp-scale-training/mlruns/{run_id}"
            f"/artifacts/model_results/{scaling_flag}/{generic_flag}/baseline_dataset_features.csv"
        )

        # Append model details to the model_records list
        model_records.append(
            {
                "run_id": pipeline_run_id,
                "model_name": model_name,
                "version_number": version_number,
                "baseline_dataset_path": baseline_dataset_path,
            }
        )

        return True

    def set_registry_uri(self, registry_uri: str):
        """
        Set the Model registery URI.

        Args:
            registry_uri (str): Model Registry URI.
        """

        return mlflow.set_registry_uri(registry_uri)

    def close(self):
        """
        Closes the current active MLflow run (if any).

        Returns:
            MLflowRun: The current instance.
        """

        if self.run:
            mlflow.end_run()  # Signal run completion in MLflow
        self.run = None
        return self

    def get_artifact_uri(self, artifact_name: str = None):
        """
        Get the artifact uri from mlflow.

        Returns:
            str: Artifact URI
        """
        return mlflow.get_artifact_uri(artifact_name)

    def add_tags(self, exp_tags: dict):
        """
        Adds the Experiment tags.

        Args:
            exp_tags (dict): Experiment tags as key-value pairs.
        Returns:
            str: Artifact URI
        """
        return mlflow.set_tag("run_id", exp_tags)

    def save_and_log_model(
        self,
        scaler_obj: object,
        trained_model: XGBRegressor,
        scaler_xgb_model: object,
        signature: mlflow.models.ModelSignature,
        model_name: str,
        model_path: str,
    ):
        """
        Save and log the model artifacts to MLflow.

        Args:
            scaler_obj (object): The StandardScaler object to be saved.
            trained_model (XGBRegressor): The trained XGBoost model to be saved.
            scaler_xgb_model (object): The custom pyfunc model
            that combines scaler and XGBoost model.
            signature (mlflow.models.ModelSignature): The model signature
            to be used for logging.
            model_name (str): The name for the model in MLflow.
            model_path (str): The base path where model artifacts will be saved.
        """
        if self.run is not None:
            # Save the StandardScaler artifact
            scaler_path = model_path + "/scaler_artifact"
            self._save_artifact(scaler_obj, scaler_path, signature, "sklearn")
            scaler_path = str(Path(scaler_path))
            # Save the XGBoost model artifact
            xgb_model_path = model_path + "/xgb_model_artifact"
            self._save_artifact(trained_model, xgb_model_path, signature, "xgboost")
            xgb_model_path = str(Path(xgb_model_path))
            # Create the dictionary of artifacts
            artifacts = {
                "scaler": scaler_path,
                "xgb_model": xgb_model_path,
            }
            print(f"artifact:{artifacts}")

            # Log the custom pyfunc model with the artifacts
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=scaler_xgb_model,
                artifacts=artifacts,
                conda_env=mlflow.sklearn.get_default_conda_env(),
                signature=signature,
            )
        else:
            print(
                """ No active run.
                  Please start a run before saving and logging the model."""
            )

    def _save_artifact(
        self,
        model_obj: object,
        artifact_path: str,
        signature: mlflow.models.ModelSignature,
        model_type: str,
    ):
        """
        Save a model artifact to the specified path.

        Args:
            model_obj (object): The model object to be saved.
            artifact_path (str): The path where the model artifact will be saved.
            model_type (str): The type of the model ('sklearn' or 'xgboost').

        Raises:
            ValueError: If the model_type is not recognized.
        """
        if self.run is not None:
            # Remove existing path if it exists
            if os.path.exists(artifact_path):
                shutil.rmtree(artifact_path)
            # Save the model artifact
            if model_type == "sklearn":
                mlflow.sklearn.save_model(model_obj, artifact_path, signature=signature)
            elif model_type == "xgboost":
                mlflow.xgboost.save_model(model_obj, artifact_path, signature=signature)

        else:
            print("No active run. Please start a run before saving artifacts.")

    def load_pyfunc_model(self, model_uri: str):
        """
        Load a model from MLflow using its URI.

        Args:
            model_uri (str): The URI of the model to be loaded.

        Returns:
            object: The loaded model if an active run exists, otherwise None.
        """
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"Error loading model from URI '{model_uri}': {e}")
            return None

    def get_tracking_uri(self):
        """
        Get the current MLflow tracking URI.

        Returns:
            str: The current MLflow tracking URI.
        """
        return mlflow.get_tracking_uri()

    def set_tracking_uri(self, tracking_uri: str):
        """
        Set the MLflow tracking URI to a file-based location.

        This method sets the tracking URI to a
        local file path where MLflow logs and stores metadata.
        """
        mlflow.set_tracking_uri(tracking_uri)

    def search_model_versions(self, model_name: str):
        """
        Searches all the model versions for a given model name.
        """
        client = MlflowClient()
        return client.search_model_versions(f"name='{model_name}'")

    def get_model_version_by_alias(self, model_name: str, model_alias: str):
        """
        Get the model version by using alias
        """
        client = MlflowClient()
        return client.get_model_version_by_alias(name=model_name, alias=model_alias)
