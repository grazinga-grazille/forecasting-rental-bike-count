"""Promote a registered model version to the champion alias after an MAE gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from forecasting_rental_bike_count.mlflow_utils import (
    configure_mlflow,
    get_latest_registered_version,
    get_model_version_mae,
    promote_model_version,
)


def _load_params(project_path: Path) -> dict:
    params_path = project_path / "conf" / "base" / "parameters.yml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def run_promote(version: str | None = None) -> str:
    """Gate MAE and set the champion alias for a registered model version.

    Args:
        version: Explicit version to promote. Defaults to the latest registered
            version for ``mlflow.registered_model_name``.

    Returns:
        The promoted version string.

    Raises:
        ValueError: If metrics are missing or the MAE gate fails.
    """
    project_path = Path(__file__).resolve().parent.parent
    params = _load_params(project_path)
    mlflow_params = params["mlflow"]

    configure_mlflow(
        tracking_uri=mlflow_params["tracking_uri"],
        experiment_name=mlflow_params["experiment_name"],
    )

    model_name = mlflow_params["registered_model_name"]
    alias = str(mlflow_params.get("champion_alias", "champion"))
    threshold = float(mlflow_params["mae_gate_threshold"])

    target_version = version or get_latest_registered_version(model_name)
    mae = get_model_version_mae(model_name, target_version)

    promoted = promote_model_version(
        model_name,
        target_version,
        mae=mae,
        mae_gate_threshold=threshold,
        alias=alias,
    )
    return promoted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a registered bike-demand model to @champion if MAE "
            "passes the configured gate."
        )
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Model version to promote (default: latest registered version).",
    )
    args = parser.parse_args(argv)

    try:
        promoted = run_promote(version=args.version)
    except ValueError as exc:
        print(f"Promote failed: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    print(f"Promoted version {promoted} to @champion")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
