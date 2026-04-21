"""
Project pipelines.
"""

from .pipelines.feature_eng import create_feature_eng_pipeline
from .pipelines.training import create_training_pipeline
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_eng_pipeline = create_feature_eng_pipeline()
    training_pipeline = create_training_pipeline()
    return {
        "__default__": feature_eng_pipeline + training_pipeline,
        "feature_eng": feature_eng_pipeline,
        "training": feature_eng_pipeline + training_pipeline,
    }
