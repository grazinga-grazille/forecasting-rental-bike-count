# Forecasting Rental Bike Count

[![Live Demo](https://img.shields.io/badge/Live-Demo-blue)](http://67.205.148.207:8050)

A machine learning pipeline for forecasting hourly rental bike demand. Built with [Kedro](https://kedro.org/) for reproducible workflows and [MLflow](https://mlflow.org/) for experiment tracking and model registry.

## Overview

This project predicts the number of rental bikes needed in the next hour using historical usage data enriched with weather, time-of-day, and seasonal features. Training registers candidate models in MLflow; a gated promote step sets the `@champion` alias; inference loads that champion for serving.

## Model lifecycle

```text
train  →  register version N  →  MAE gate + promote (@champion)  →  inference serves @champion
```

| Step | Command | What happens |
|---|---|---|
| Train | `kedro run --pipeline=training` | Fits model, logs params/metrics, **registers** a new version (does not go live) |
| Promote | `python entrypoints/promote.py` | Reads run MAE; if ≤ `mae_gate_threshold`, sets `@champion` |
| Infer | `kedro run --pipeline=inference` | Loads `models:/bike_demand_forecast@champion` when `model_storage.source: mlflow` |

Optional version pin:

```bash
python entrypoints/promote.py --version 2
```

### Local vs MLflow serving

In `conf/base/parameters.yml`:

```yaml
model_storage:
  source: mlflow  # or local
  path: data/06_models
  name: forecast_model
```

- **`mlflow`** — inference loads the registry alias `@champion` (falls back to local `.cbm`/`.pkl` if the alias is missing)
- **`local`** — inference loads `data/06_models/forecast_model.cbm` (or `.pkl`)

Training always saves a local copy under `data/06_models/` as an offline cache, and logs to MLflow when configured.

### Rollback

Re-point `@champion` at a previous good version (no code change):

```bash
python entrypoints/promote.py --version 1
```

Inference keeps using `models:/bike_demand_forecast@champion`; only the alias target changes.

### MLflow UI

Shared store: `sqlite:///mlruns/mlflow.db` (directory `./mlruns`, gitignored except `.gitkeep`).

**Docker Compose** (host port **5001** → container 5000):

```bash
docker compose up mlflow
# open http://localhost:5001
```

**Local:**

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
# open http://localhost:5000
```

Override the tracking URI without editing YAML:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
```

## Project Structure

```
forecasting-rental-bike-count/
├── conf/base/
│   ├── catalog.yml
│   └── parameters.yml          # features, model, model_storage, mlflow
├── data/                       # Parquet layers + local model cache
├── mlruns/                     # MLflow DB + artifacts (Compose volume)
├── entrypoints/
│   ├── training.py
│   ├── promote.py
│   ├── check_mae_gate.py
│   ├── inference.py
│   └── app_ui.py
├── notebooks/Modeling.ipynb
├── .github/workflows/
│   ├── ci.yml / cd.yml
│   ├── train.yml / promote.yml / inference.yml
├── src/forecasting_rental_bike_count/
│   ├── mlflow_utils.py
│   ├── pipelines/
│   └── ...
├── docker-compose.yml
└── pyproject.toml
```

## Pipelines

### Feature engineering

1. **Rename columns** — e.g. `hr` → `hour`, `cnt` → `bike_count`
2. **Lag features** — for `bike_count`, `hour`, `temperature`, `humidity`

### Training (`training`)

1. Make next-hour target  
2. Chronological train/test split  
3. Train selected model  
4. Predict + compute MAE / RMSE / MAPE  
5. Save local model under `data/06_models/`  
6. Log to MLflow and register a model version (no `@champion` yet)

### Inference (`inference`)

1. Load model (`mlflow` `@champion` or `local` file)  
2. Predict on the inference batch  
3. Write `data/07_model_output/predictions.parquet`

## Supported Models

| Model | Key | Notes |
|---|---|---|
| CatBoost | `catboost` / `cb` | Primary; local save as `.cbm` |
| Random Forest | `random_forest` / `rf` | `.pkl` via joblib |
| Linear Regression | `linear_regression` / `linreg` | `.pkl` via joblib |

Active model: `training.model_type` in `parameters.yml`.

## Quickstart

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) (recommended)

### Installation

```bash
git clone <repo-url>
cd forecasting-rental-bike-count
uv sync
# or: pip install -e ".[dev]"
```

### Data

Place training / inference parquet under `data/01_raw/` (e.g. `bike_data_train.parquet`, `bike_data_inference.parquet`) with at least: `datetime`, `season`, `hr`, `weekday`, `weathersit`, `temp`, `hum`, `windspeed`, `cnt`.

### Train → promote → infer (local)

```bash
uv run kedro run --pipeline=training
uv run python entrypoints/promote.py
uv run kedro run --pipeline=inference
```

### Docker Compose

Train and inference share `./mlruns` with the MLflow service:

```bash
docker compose build
docker compose up mlflow          # UI on http://localhost:5001
docker compose run --rm ml-train
uv run python entrypoints/promote.py   # on host against ./mlruns
docker compose up ml-inference app-ui  # Dash on http://localhost:8050
```

### Tests and lint

```bash
uv run pytest
uv run ruff check src/ tests/
```

### Pipeline viz

```bash
uv run kedro viz
```

## GitHub Actions

| Workflow | Trigger | Role |
|---|---|---|
| `ci.yml` | PR / push to `main` | Lint, pytest, Docker build |
| `cd.yml` | Push to `main` | Publish image to GHCR |
| `train.yml` | Manual + weekly cron | Train, MAE gate, upload `mlruns` artifact |
| `promote.yml` | Manual | Download train artifact, promote `@champion` |
| `inference.yml` | Manual | Smoke inference against promoted (or train) artifact |

Typical Actions order: **Train** → **Promote** → **Inference**. Runners are ephemeral; the registry is passed between jobs as the `mlruns` artifact.

## Configuration

Key blocks in `conf/base/parameters.yml`:

```yaml
training:
  model_type: catboost
  train_fraction: 0.8
  model_params:
    catboost: { ... }

model_storage:
  source: mlflow          # local | mlflow
  path: data/06_models
  name: forecast_model

mlflow:
  tracking_uri: sqlite:///mlruns/mlflow.db
  experiment_name: bike_demand_forecast
  registered_model_name: bike_demand_forecast
  register_on_train: true   # register only; promote sets @champion
  champion_alias: champion
  mae_gate_threshold: 50
```

## Tech Stack

- **[Kedro](https://kedro.org/)** — pipeline orchestration
- **[MLflow](https://mlflow.org/)** — tracking + model registry (`@champion`)
- **[CatBoost](https://catboost.ai/)** — primary model
- **[scikit-learn](https://scikit-learn.org/)** — RF / linear baselines
- **[Optuna](https://optuna.org/)** — hyperparameter search (notebook)
- **[Plotly Dash](https://dash.plotly.com/)** — live demo UI
- **[Docker Compose](https://docs.docker.com/compose/)** — train / inference / MLflow / UI services
- **[ruff](https://docs.astral.sh/ruff/)** — linting
