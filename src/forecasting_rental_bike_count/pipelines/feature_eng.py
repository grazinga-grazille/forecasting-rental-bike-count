from kedro.pipeline import node, Pipeline
from .nodes import rename_columns, create_lag_features

def create_feature_eng_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=rename_columns,
            inputs=["train_data", "params:feature_engineering.rename_columns"],
            outputs="renamed_data",
        ),
        node(
            func=create_lag_features,
            inputs=["renamed_data", "params:feature_engineering.lag_params"],
            outputs=["features", "timestamps"],
        )
    ])