from kedro.pipeline import Pipeline

from .pipelines.feature_eng import (
    feat_eng_pipeline_inference,
    feat_eng_pipeline_training,
)
from .pipelines.inference import create_inference_pipeline
from .pipelines.training import create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    feature_eng_training = feat_eng_pipeline_training()
    feature_eng_inference = feat_eng_pipeline_inference()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    return {
        "__default__": feature_eng_training + training_pipeline,
        "training": feature_eng_training + training_pipeline,
        "inference": feature_eng_inference + inference_pipeline,
    }
