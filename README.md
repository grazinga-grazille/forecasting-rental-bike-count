# Forecasting Rental Bike Count

A machine learning pipeline for forecasting hourly rental bike demand. Built with [Kedro](https://kedro.org/) for reproducible, modular ML workflows.

## Overview

This project predicts the number of rental bikes needed in the next hour using historical usage data enriched with weather, time-of-day, and seasonal features. It supports multiple regression models and is structured as a production-ready Kedro pipeline.

## Project Structure

```
forecasting-rental-bike-count/
├── conf/
│   ├── base/
│   │   ├── catalog.yml        # Dataset definitions
│   │   └── parameters.yml     # Pipeline parameters (features, model config)
│   └── local/                 # Local overrides (credentials, etc.)
├── data/
│   ├── 01_raw/                # Raw input data (Parquet)
│   ├── 06_models/             # Serialized trained models
│   └── ...                    # Intermediate Kedro data layers
├── notebooks/
│   └── Modeling.ipynb         # Exploratory modeling notebook
├── src/
│   └── forecasting_rental_bike_count/
│       ├── pipelines/
│       │   ├── feature_eng.py # Feature engineering pipeline
│       │   ├── training.py    # Training pipeline
│       │   └── nodes.py       # All pipeline node functions
│       ├── pipeline_registry.py
│       └── settings.py
└── pyproject.toml
```

## Pipelines

### Feature Engineering (`feature_eng`)
1. **Rename columns** — maps raw dataset column names to human-readable names (e.g., `hr` → `hour`, `cnt` → `bike_count`)
2. **Create lag features** — generates lag features for `bike_count`, `hour`, `temperature`, and `humidity` to capture temporal dependencies

### Training (`training`)
1. **Make target** — shifts `bike_count` by one period to create a next-hour forecast target
2. **Split data** — chronological 80/20 train-test split
3. **Train model** — fits the selected regression model
4. **Predict** — generates predictions on the test set
5. **Compute metrics** — evaluates with MAE, RMSE, and MAPE
6. **Save model** — persists the trained model to `data/06_models/`

## Supported Models

| Model | Key | Notes |
|---|---|---|
| CatBoost | `catboost` / `cb` | Saved as `.cbm` (native format) |
| Random Forest | `random_forest` / `rf` | Saved as `.pkl` via joblib |
| Linear Regression | `linear_regression` / `linreg` | Saved as `.pkl` via joblib |

The active model is set in `conf/base/parameters.yml` under `training.model_type`.

## Quickstart

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd forecasting-rental-bike-count

# Install dependencies
uv sync
# or: pip install -e ".[dev]"
```

### Add Data

Place the raw training data at:

```
data/01_raw/bike_data_train.parquet
```

The dataset should contain at minimum: `datetime`, `season`, `hr`, `weekday`, `weathersit`, `temp`, `hum`, `windspeed`, `cnt`.

### Run the Pipeline

```bash
# Run the full pipeline (feature engineering + training)
kedro run

# Run only feature engineering
kedro run --pipeline feature_eng

# Run feature engineering + training
kedro run --pipeline training
```

### Visualize the Pipeline

```bash
kedro viz
```

### Run Tests

```bash
pytest
```

### Lint

```bash
ruff check src/
```

## Configuration

All pipeline behaviour is controlled via `conf/base/parameters.yml`:

```yaml
feature_engineering:
  rename_columns:          # Column rename mapping
    ...
  lag_params:              # Lag windows per feature
    bike_count: [1, 2, 22, 23]
    hour: [1, 2, 3]
    temperature: [1, 2, 3]
    humidity: [1, 2, 3]

training:
  target_params:
    shift_period: 1        # Forecast horizon (hours)
    target_column: bike_count
  train_fraction: 0.8      # Train/test split ratio
  model_type: catboost     # Active model
  model_params:
    catboost:
      learning_rate: 0.2
      depth: 6
      iterations: 50
      loss_function: RMSE
      ...

model_storage:
  path: data/06_models
  name: forecast_model
```

## Tech Stack

- **[Kedro](https://kedro.org/)** — pipeline orchestration and project structure
- **[CatBoost](https://catboost.ai/)** — gradient boosting (primary model)
- **[scikit-learn](https://scikit-learn.org/)** — Random Forest and Linear Regression
- **[pandas](https://pandas.pydata.org/)** / **[NumPy](https://numpy.org/)** — data manipulation
- **[Plotly Dash](https://dash.plotly.com/)** — interactive visualizations
- **[Kedro-Viz](https://github.com/kedro-org/kedro-viz)** — pipeline visualization
- **[ruff](https://docs.astral.sh/ruff/)** — linting and formatting
