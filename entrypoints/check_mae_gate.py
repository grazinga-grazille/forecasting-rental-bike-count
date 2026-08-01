"""Fail if the latest registered model's MAE exceeds the configured gate."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from forecasting_rental_bike_count.mlflow_utils import (
    configure_mlflow,
    get_latest_registered_version,
    get_model_version_mae,
)


def main() -> int:
    project_path = Path(__file__).resolve().parent.parent
    with open(project_path / "conf" / "base" / "parameters.yml") as f:
        params = yaml.safe_load(f)

    mlflow_params = params["mlflow"]
    configure_mlflow(
        tracking_uri=mlflow_params["tracking_uri"],
        experiment_name=mlflow_params["experiment_name"],
    )

    model_name = mlflow_params["registered_model_name"]
    threshold = float(mlflow_params["mae_gate_threshold"])
    version = get_latest_registered_version(model_name)
    mae = get_model_version_mae(model_name, version)

    if mae > threshold:
        print(  # noqa: T201
            f"MAE gate failed: mae={mae} > threshold={threshold} "
            f"(model={model_name} v{version})",
            file=sys.stderr,
        )
        return 1

    print(  # noqa: T201
        f"MAE gate passed: mae={mae} <= threshold={threshold} "
        f"(model={model_name} v{version})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
