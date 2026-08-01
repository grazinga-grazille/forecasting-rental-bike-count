"""MLflow helpers for experiment tracking and model registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Set the tracking URI and ensure the experiment exists.

    ``MLFLOW_TRACKING_URI`` overrides ``tracking_uri`` when set (Compose/Actions).

    Prefer a DB-backed URI for local work, e.g. ``sqlite:///mlruns/mlflow.db``.
    File-store URIs (``file:./mlruns``) still work if opted in via env.
    """
    effective_uri = os.environ.get("MLFLOW_TRACKING_URI") or tracking_uri

    if effective_uri.startswith("file:"):
        # MLflow 3+ disables the filesystem backend unless explicitly allowed.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    elif effective_uri.startswith("sqlite:"):
        # Ensure parent directory exists for sqlite:///path/to/file.db
        db_path = effective_uri.removeprefix("sqlite:///")
        if db_path and not db_path.startswith(":"):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(effective_uri)
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
) -> tuple[str, str | None]:
    """Log params, metrics, and model artifact for a training run.

    Registers a model version when ``register_on_train`` is True. Does **not**
    set the champion alias — promotion is a separate gated step.

    Args:
        params: Flat or nested-ish training parameters to log.
        metrics: Evaluation metrics (e.g. MAE, RMSE, MAPE).
        model: Trained model instance to log as an artifact.
        model_type: Model family key (catboost, random_forest, ...).
        registered_model_name: Registry name when register_on_train is True.
        register_on_train: If True, register the logged model version.
        run_name: Optional MLflow run display name.

    Returns:
        ``(run_id, registered_version)`` where ``registered_version`` is
        ``None`` when registration was skipped.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_params(params))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("stage", "train")

        model_info = _log_model(model, model_type)

        registered_version: str | None = None
        if register_on_train and registered_model_name:
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=registered_model_name,
            )
            registered_version = str(model_version.version)
            mlflow.set_tag("registered_version", registered_version)
            mlflow.set_tag("gated", "false")

        return run.info.run_id, registered_version


def _log_model(model: Any, model_type: str) -> Any:
    """Log a model with the appropriate MLflow flavor."""
    normalized = model_type.lower().strip()
    if normalized in {"catboost", "cb"}:
        model_info = mlflow.catboost.log_model(model, artifact_path="model")
    elif normalized in {"random_forest", "rf", "linear_regression", "linreg"}:
        model_info = mlflow.sklearn.log_model(model, artifact_path="model")
    else:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_UnsupportedModelWrapper(model),
        )
    mlflow.flush()
    return model_info


def get_champion_model_uri(
    registered_model_name: str,
    alias: str = "champion",
) -> str:
    """Return the MLflow URI for a registered model alias."""
    return f"models:/{registered_model_name}@{alias}"


def load_champion_model(
    registered_model_name: str,
    alias: str = "champion",
) -> Any:
    """Load the model version pointed to by ``alias`` (default: champion).

    Raises:
        ValueError: If the alias is not set on the registered model.
    """
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(registered_model_name, alias)
    except MlflowException as exc:
        raise ValueError(
            f"No '{alias}' alias found for registered model "
            f"'{registered_model_name}'. Train a model, then run "
            f"`python entrypoints/promote.py` (or temporarily set "
            f"model_storage.source: local)."
        ) from exc

    model_uri = get_champion_model_uri(registered_model_name, alias=alias)
    return mlflow.pyfunc.load_model(model_uri)


def get_latest_registered_version(registered_model_name: str) -> str:
    """Return the highest version number for a registered model."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    if not versions:
        raise ValueError(
            f"No registered versions found for '{registered_model_name}'. "
            "Run the training pipeline first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return str(latest.version)


def get_model_version_mae(
    registered_model_name: str,
    version: str | int,
    metric_key: str = "MAE",
) -> float:
    """Read a holdout metric from the training run linked to a model version."""
    client = MlflowClient()
    model_version = client.get_model_version(
        registered_model_name, str(version)
    )
    if not model_version.run_id:
        raise ValueError(
            f"Model '{registered_model_name}' version {version} has no "
            "linked MLflow run; cannot read metrics for the gate."
        )
    run = client.get_run(model_version.run_id)
    if metric_key not in run.data.metrics:
        raise ValueError(
            f"Metric '{metric_key}' not found on run {model_version.run_id} "
            f"for '{registered_model_name}' v{version}."
        )
    return float(run.data.metrics[metric_key])


def promote_model_version(  # noqa: PLR0913
    registered_model_name: str,
    version: str | int,
    *,
    mae: float,
    mae_gate_threshold: float,
    alias: str = "champion",
) -> str:
    """Set ``alias`` on ``version`` if MAE passes the quality gate.

    Args:
        registered_model_name: Registry model name.
        version: Model version to promote.
        mae: Holdout MAE for this version's training run.
        mae_gate_threshold: Maximum allowed MAE (inclusive).
        alias: Alias to assign (default: champion).

    Returns:
        The promoted version as a string.

    Raises:
        ValueError: If ``mae`` exceeds ``mae_gate_threshold``.
    """
    version_str = str(version)
    if mae > mae_gate_threshold:
        raise ValueError(
            f"MAE gate failed: mae={mae} > threshold={mae_gate_threshold}. "
            f"Version {version_str} of '{registered_model_name}' was not "
            f"promoted to '{alias}'."
        )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version_str,
    )
    return version_str


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
