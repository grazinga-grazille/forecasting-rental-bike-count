"""Tests for dual-path model loading and predict normalization."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from forecasting_rental_bike_count.pipelines.nodes import load_model, predict

MLFLOW_PARAMS = {
    "tracking_uri": "sqlite:///mlruns/mlflow.db",
    "experiment_name": "bike_demand_forecast",
    "registered_model_name": "bike_demand_forecast",
    "champion_alias": "champion",
}


def test_load_model_local_joblib(tmp_path):
    model = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])
    joblib.dump(model, tmp_path / "forecast_model.pkl")

    loaded = load_model(
        "linear_regression",
        {
            "source": "local",
            "path": str(tmp_path),
            "name": "forecast_model",
        },
        mlflow_params=MLFLOW_PARAMS,
    )

    pred = loaded.predict([[3.0]])
    assert float(np.asarray(pred).ravel()[0]) == pytest.approx(3.0)


def test_load_model_mlflow_uses_champion_helper(mocker):
    configure = mocker.patch(
        "forecasting_rental_bike_count.pipelines.nodes.configure_mlflow"
    )
    champion = object()
    load_champion = mocker.patch(
        "forecasting_rental_bike_count.pipelines.nodes.load_champion_model",
        return_value=champion,
    )

    result = load_model(
        "catboost",
        {"source": "mlflow", "path": "unused", "name": "unused"},
        mlflow_params=MLFLOW_PARAMS,
    )

    assert result is champion
    configure.assert_called_once_with(
        tracking_uri=MLFLOW_PARAMS["tracking_uri"],
        experiment_name=MLFLOW_PARAMS["experiment_name"],
    )
    load_champion.assert_called_once_with(
        registered_model_name="bike_demand_forecast",
        alias="champion",
    )


def test_load_model_mlflow_requires_params():
    with pytest.raises(ValueError, match="params:mlflow is required"):
        load_model(
            "catboost",
            {"source": "mlflow"},
            mlflow_params=None,
        )


def test_load_model_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown model_storage.source"):
        load_model(
            "catboost",
            {"source": "s3"},
            mlflow_params=MLFLOW_PARAMS,
        )


def test_load_model_mlflow_missing_champion_falls_back_to_local(mocker, tmp_path):
    model = LinearRegression().fit([[1.0], [2.0]], [1.0, 2.0])
    joblib.dump(model, tmp_path / "forecast_model.pkl")

    mocker.patch(
        "forecasting_rental_bike_count.pipelines.nodes.configure_mlflow"
    )
    mocker.patch(
        "forecasting_rental_bike_count.pipelines.nodes.load_champion_model",
        side_effect=ValueError(
            "No 'champion' alias found for registered model "
            "'bike_demand_forecast'."
        ),
    )

    loaded = load_model(
        "linear_regression",
        {
            "source": "mlflow",
            "path": str(tmp_path),
            "name": "forecast_model",
        },
        mlflow_params=MLFLOW_PARAMS,
    )

    pred = loaded.predict([[3.0]])
    assert float(np.asarray(pred).ravel()[0]) == pytest.approx(3.0)


def test_predict_normalizes_numpy_column_vector():
    class FakeModel:
        def predict(self, x):
            return np.array([[1.5], [2.5]])

    out = predict(FakeModel(), pd.DataFrame({"a": [0, 1]}))
    assert list(out.columns) == ["prediction"]
    assert list(out["prediction"]) == [1.5, 2.5]


def test_predict_normalizes_dataframe_output():
    class FakeModel:
        def predict(self, x):
            return pd.DataFrame({"prediction": [10.0, 20.0]})

    out = predict(FakeModel(), pd.DataFrame({"a": [0, 1]}))
    assert list(out["prediction"]) == [10.0, 20.0]
