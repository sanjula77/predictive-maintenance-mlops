"""
MLflow utilities for experiment tracking and model registry.

This module provides functions to:
- Initialize MLflow experiments
- Log training metrics and parameters
- Register models in MLflow Model Registry
- Load models from registry for production
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "Predictive-Maintenance-RUL"
MODEL_REGISTRY_NAME = "RUL-Prediction-Model"


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None) -> None:
    """Initialize MLflow tracking.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking URI. If None, uses local file system.
    experiment_name : str, optional
        Experiment name. If None, uses default.
    """
    # Set tracking URI
    if tracking_uri is None:
        # Convert Path to proper file URI for Windows compatibility
        # Use as_uri() to handle Windows paths correctly (e.g., C:\path -> file:///C:/path)
        tracking_uri = MLFLOW_TRACKING_URI.as_uri()
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment
    if experiment_name is None:
        experiment_name = MLFLOW_EXPERIMENT_NAME

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})"
            )
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not set up MLflow experiment: {e}")
        logger.info("MLflow tracking will be disabled")


def log_training_params(
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    seed: int,
    additional_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Log training hyperparameters to MLflow.

    Parameters
    ----------
    model_type : str
        Model architecture type (lstm, bilstm, gru, transformer).
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate.
    sequence_length : int
        Sequence length for time series.
    hidden_size : int
        Hidden layer size.
    num_layers : int
        Number of layers.
    dropout : float
        Dropout rate.
    seed : int
        Random seed.
    additional_params : dict, optional
        Additional parameters to log.
    """
    params = {
        "model_type": model_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "seed": seed,
    }

    if additional_params:
        params.update(additional_params)

    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} parameters to MLflow")


def log_training_metrics(
    train_loss: float,
    val_loss: Optional[float] = None,
    test_rmse: Optional[float] = None,
    test_mae: Optional[float] = None,
    epoch: Optional[int] = None,
) -> None:
    """Log training metrics to MLflow.

    Parameters
    ----------
    train_loss : float
        Training loss.
    val_loss : float, optional
        Validation loss.
    test_rmse : float, optional
        Test RMSE.
    test_mae : float, optional
        Test MAE.
    epoch : int, optional
        Epoch number for step logging.
    """
    metrics = {"train_loss": train_loss}

    if val_loss is not None:
        metrics["val_loss"] = val_loss
    if test_rmse is not None:
        metrics["test_rmse"] = test_rmse
    if test_mae is not None:
        metrics["test_mae"] = test_mae

    if epoch is not None:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)
    else:
        mlflow.log_metrics(metrics)

    logger.debug(f"Logged metrics to MLflow: {metrics}")


def log_model_artifacts(
    model: Any,
    scaler: Any,
    model_type: str,
    scaler_path: Optional[Path] = None,
) -> None:
    """Log model and scaler artifacts to MLflow.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler.
    model_type : str
        Model type for naming.
    scaler_path : Path, optional
        Path to save scaler. If None, saves to temp location.
    """
    # Log PyTorch model
    mlflow.pytorch.log_model(model, "model")
    logger.info("Logged PyTorch model to MLflow")

    # Log scaler
    if scaler_path is None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            import joblib

            joblib.dump(scaler, tmp.name)
            scaler_path = Path(tmp.name)

    mlflow.log_artifact(str(scaler_path), "scaler")
    logger.info("Logged scaler to MLflow")


def register_model_in_mlflow(
    run_id: str,
    model_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
) -> str:
    """Register model in MLflow Model Registry.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    model_name : str, optional
        Model name in registry. If None, uses default.
    tags : dict, optional
        Tags to add to model version.
    description : str, optional
        Model description.

    Returns
    -------
    str
        Model version.
    """
    if model_name is None:
        model_name = MODEL_REGISTRY_NAME

    model_uri = f"runs:/{run_id}/model"

    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(f"Registered model version {model_version.version} in MLflow registry")

        # Add tags if provided
        if tags:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name, version=model_version.version, key=key, value=str(value)
                )

        # Add description if provided
        if description:
            client = MlflowClient()
            client.update_model_version(
                name=model_name, version=model_version.version, description=description
            )

        return model_version.version
    except Exception as e:
        logger.error(f"Failed to register model in MLflow: {e}")
        raise


def get_production_model(model_name: Optional[str] = None) -> Any:
    """Load production model from MLflow Model Registry using alias.

    Parameters
    ----------
    model_name : str, optional
        Model name in registry. If None, uses default.

    Returns
    -------
    torch.nn.Module
        Loaded PyTorch model.
    """
    if model_name is None:
        model_name = MODEL_REGISTRY_NAME

    try:
        # Use alias instead of stage (new MLflow UI)
        model_uri = f"models:/{model_name}@production"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded production model from MLflow (using alias): {model_name}")
        return model
    except Exception as e:
        logger.warning(f"Could not load production model using alias: {e}")
        logger.info("Falling back to latest model version")
        try:
            client = MlflowClient()
            # Get latest version without stage filter
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                model = mlflow.pytorch.load_model(model_uri)
                logger.info(f"Loaded latest model version {latest_version.version}")
                return model
            else:
                raise ValueError(f"No versions found for model {model_name}")
        except Exception as e2:
            logger.error(f"Failed to load model from MLflow: {e2}")
            raise


def get_staging_model(model_name: Optional[str] = None) -> Any:
    """Load staging model from MLflow Model Registry.

    Parameters
    ----------
    model_name : str, optional
        Model name in registry. If None, uses default.

    Returns
    -------
    torch.nn.Module
        Loaded PyTorch model.
    """
    if model_name is None:
        model_name = MODEL_REGISTRY_NAME

    try:
        model_uri = f"models:/{model_name}/Staging"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Loaded staging model from MLflow: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Could not load staging model from MLflow: {e}")
        raise


def promote_model_to_staging(model_name: str, version: int) -> None:
    """Promote model version to Staging stage.

    Parameters
    ----------
    model_name : str
        Model name in registry.
    version : int
        Model version number.
    """
    client = MlflowClient()
    client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
    logger.info(f"Promoted model {model_name} version {version} to Staging")


def promote_model_to_production(model_name: str, version: int) -> None:
    """Promote model version to Production using alias (new MLflow UI).

    Parameters
    ----------
    model_name : str
        Model name in registry.
    version : int
        Model version number.
    """
    client = MlflowClient()

    # Remove alias from current production model if exists
    try:
        # Get model info to check for existing production alias
        model_info = client.get_registered_model(name=model_name)
        aliases_dict = getattr(model_info, "aliases", {}) or {}

        # If production alias exists, delete it first (aliases are unique)
        if "production" in aliases_dict:
            client.delete_registered_model_alias(name=model_name, alias="production")
            logger.info(f"Removed production alias from version {aliases_dict['production']}")
    except Exception as e:
        logger.debug(f"No existing production alias to remove: {e}")

    # Set alias for new model version
    client.set_registered_model_alias(name=model_name, alias="production", version=str(version))
    logger.info(f"Promoted model {model_name} version {version} to Production (using alias)")


def get_best_model_by_metric(
    metric_name: str = "test_rmse", ascending: bool = True
) -> Optional[Dict[str, Any]]:
    """Find best model run by metric.

    Parameters
    ----------
    metric_name : str
        Metric name to search for (default: "test_rmse").
    ascending : bool
        If True, lower is better (for RMSE, MAE). If False, higher is better.

    Returns
    -------
    dict or None
        Best run information, or None if no runs found.
    """
    try:
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if experiment is None:
            logger.warning("No MLflow experiment found")
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if not runs:
            logger.warning("No runs found in MLflow experiment")
            return None

        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "metric_value": best_run.data.metrics.get(metric_name),
            "model_type": best_run.data.params.get("model_type"),
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
        }
    except Exception as e:
        logger.error(f"Error finding best model: {e}")
        return None


def list_registered_models(model_name: Optional[str] = None) -> list:
    """List all registered model versions with aliases.

    Parameters
    ----------
    model_name : str, optional
        Model name in registry. If None, uses default.

    Returns
    -------
    list
        List of model version dictionaries with alias information.
    """
    if model_name is None:
        model_name = MODEL_REGISTRY_NAME

    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        model_list = []
        # Get all aliases for the registered model
        model_aliases = {}
        try:
            model_info = client.get_registered_model(name=model_name)
            # MLflow returns aliases as a dict: {alias_name: version}
            aliases_dict = getattr(model_info, "aliases", {}) or {}
            # Reverse mapping: {version: [alias_names]}
            for alias_name, alias_version in aliases_dict.items():
                if alias_version not in model_aliases:
                    model_aliases[alias_version] = []
                model_aliases[alias_version].append(alias_name)
        except Exception as e:
            logger.warning(f"Could not get aliases: {e}")

        for version in versions:
            version_str = str(version.version)
            # Get aliases for this version
            aliases = model_aliases.get(version_str, [])

            # Determine stage based on aliases (for backward compatibility)
            stage = "None"
            if "production" in aliases:
                stage = "Production"
            elif "staging" in aliases:
                stage = "Staging"

            model_list.append(
                {
                    "version": version.version,
                    "stage": stage,  # For backward compatibility
                    "aliases": aliases,  # New: actual aliases
                    "run_id": version.run_id,
                    "created_at": version.creation_timestamp,
                    "description": version.description,
                }
            )

        return sorted(model_list, key=lambda x: int(x["version"]), reverse=True)
    except Exception as e:
        logger.error(f"Error listing registered models: {e}")
        return []
