from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_metrics,
    log_model_to_mlflow,
    make_target,
    predict,
    save_model,
    split_data,
    train_model,
)


def create_training_pipeline() -> Pipeline:
    return Pipeline([
        node(
            func=make_target,
            inputs=["features", "params:training.target_params"],
            outputs="data_with_target",
        ),
        node(
            func=split_data,
            inputs=["data_with_target", "params:training"],
            outputs=["x_train", "x_test", "y_train", "y_test"],
        ),
        node(
            func=train_model,
            inputs=["x_train", "y_train", "params:training"],
            outputs="trained_model",
        ),
        node(
            func=predict,
            inputs=["trained_model", "x_test"],
            outputs="predictions",
        ),
        node(
            func=compute_metrics,
            inputs=["y_test", "predictions"],
            outputs="metrics",
        ),
        node(
            func=save_model,
            inputs=["trained_model", "params:training.model_type", "params:model_storage"],
            outputs=None,
        ),
        node(
            func=log_model_to_mlflow,
            inputs=[
                "trained_model",
                "metrics",
                "params:training",
                "params:mlflow",
            ],
            outputs="mlflow_run_id",
        ),
    ])
