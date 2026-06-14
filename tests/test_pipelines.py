from forecasting_rental_bike_count.pipeline_registry import register_pipelines


def test_register_pipelines_returns_expected_keys():
    pipelines = register_pipelines()
    assert set(pipelines) == {"__default__", "training", "inference"}


def test_default_pipeline_has_nodes():
    pipelines = register_pipelines()
    assert len(pipelines["__default__"].nodes) > 0
