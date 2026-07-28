"""MLflow helpers for experiment tracking and model registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Set the tracking URI and ensure the experiment exists.

    Prefer a DB-backed URI for local work, e.g. ``sqlite:///mlruns/mlflow.db``.
    File-store URIs (``file:./mlruns``) still work if opted in via env.
    """
    if tracking_uri.startswith("file:"):
        # MLflow 3+ disables the filesystem backend unless explicitly allowed.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    elif tracking_uri.startswith("sqlite:"):
        # Ensure parent directory exists for sqlite:///path/to/file.db
        db_path = tracking_uri.removeprefix("sqlite:///")
        if db_path and not db_path.startswith(":"):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_training_run(  # noqa: PLR0913
    *,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any,
    model_type: str,
    registered_model_name: str | None = None,
    register_on_train: bool = False,
    run_name: str | None = None,
) -> str:
    """Log params, metrics, and model artifact for a training run.

    Args:
        params: Flat or nested-ish training parameters to log.
        metrics: Evaluation metrics (e.g. MAE, RMSE, MAPE).
        model: Trained model instance to log as an artifact.
        model_type: Model family key (catboost, random_forest, ...).
        registered_model_name: Registry name when register_on_train is True.
        register_on_train: If True, register the logged model version.
        run_name: Optional MLflow run display name.

    Returns:
        The MLflow run ID.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_params(params))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", model_type)

        model_info = _log_model(model, model_type)

        if register_on_train and registered_model_name:
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name=registered_model_name,
            )

        return run.info.run_id


def _log_model(model: Any, model_type: str) -> Any:
    """Log a model with the appropriate MLflow flavor."""
    normalized = model_type.lower().strip()
    if normalized in {"catboost", "cb"}:
        return mlflow.catboost.log_model(model, name="model")
    if normalized in {"random_forest", "rf", "linear_regression", "linreg"}:
        return mlflow.sklearn.log_model(model, name="model")
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=_UnsupportedModelWrapper(model),
    )


def get_production_model_uri(registered_model_name: str) -> str:
    """Return the MLflow URI for the Production stage of a registered model."""
    return f"models:/{registered_model_name}/Production"


def load_production_model(registered_model_name: str) -> Any:
    """Load the Production version of a registered model."""
    model_uri = get_production_model_uri(registered_model_name)
    client = MlflowClient()
    versions = client.get_latest_versions(
        registered_model_name, stages=["Production"]
    )
    if not versions:
        raise ValueError(
            f"No Production version found for '{registered_model_name}'"
        )
    return mlflow.pyfunc.load_model(model_uri)


def _flatten_params(
    params: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, str | int | float | bool]:
    """Flatten nested dicts so MLflow can log them as params."""
    items: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_params(value, new_key, sep=sep))
        elif isinstance(value, str | int | float | bool):
            items[new_key] = value
        else:
            items[new_key] = str(value)
    return items


class _UnsupportedModelWrapper(mlflow.pyfunc.PythonModel):
    """Minimal pyfunc wrapper for unexpected model types."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
