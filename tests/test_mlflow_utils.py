"""Unit tests for MLflow registry helpers (aliases + MAE gate)."""

from __future__ import annotations

import pytest
from mlflow.exceptions import MlflowException

from forecasting_rental_bike_count.mlflow_utils import (
    get_champion_model_uri,
    load_champion_model,
    promote_model_version,
)


def test_get_champion_model_uri_default_alias():
    assert (
        get_champion_model_uri("bike_demand_forecast")
        == "models:/bike_demand_forecast@champion"
    )


def test_get_champion_model_uri_custom_alias():
    assert (
        get_champion_model_uri("bike_demand_forecast", alias="staging")
        == "models:/bike_demand_forecast@staging"
    )


def test_promote_rejects_mae_above_threshold(mocker):
    client = mocker.Mock()
    mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.MlflowClient",
        return_value=client,
    )

    with pytest.raises(ValueError, match="MAE gate failed"):
        promote_model_version(
            "bike_demand_forecast",
            1,
            mae=51.0,
            mae_gate_threshold=50.0,
        )

    client.set_registered_model_alias.assert_not_called()


def test_promote_sets_champion_alias_when_mae_passes(mocker):
    client = mocker.Mock()
    mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.MlflowClient",
        return_value=client,
    )

    version = promote_model_version(
        "bike_demand_forecast",
        3,
        mae=32.1,
        mae_gate_threshold=50.0,
        alias="champion",
    )

    assert version == "3"
    client.set_registered_model_alias.assert_called_once_with(
        name="bike_demand_forecast",
        alias="champion",
        version="3",
    )


def test_promote_allows_mae_equal_to_threshold(mocker):
    client = mocker.Mock()
    mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.MlflowClient",
        return_value=client,
    )

    promote_model_version(
        "bike_demand_forecast",
        1,
        mae=50.0,
        mae_gate_threshold=50.0,
    )

    client.set_registered_model_alias.assert_called_once()


def test_load_champion_model_missing_alias_raises_clear_error(mocker):
    client = mocker.Mock()
    client.get_model_version_by_alias.side_effect = MlflowException("RESOURCE_DOES_NOT_EXIST")
    mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.MlflowClient",
        return_value=client,
    )
    load_model = mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.mlflow.pyfunc.load_model"
    )

    with pytest.raises(ValueError, match="No 'champion' alias found"):
        load_champion_model("bike_demand_forecast")

    load_model.assert_not_called()


def test_load_champion_model_loads_alias_uri(mocker):
    client = mocker.Mock()
    client.get_model_version_by_alias.return_value = mocker.Mock(version="1")
    mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.MlflowClient",
        return_value=client,
    )
    loaded = object()
    load_model = mocker.patch(
        "forecasting_rental_bike_count.mlflow_utils.mlflow.pyfunc.load_model",
        return_value=loaded,
    )

    result = load_champion_model("bike_demand_forecast", alias="champion")

    assert result is loaded
    load_model.assert_called_once_with("models:/bike_demand_forecast@champion")
