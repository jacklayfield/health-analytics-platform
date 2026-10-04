from types import SimpleNamespace

import pytest

from src.serving import model_server


@pytest.mark.parametrize(
    ("model_uri", "model_key", "expected_version"),
    [
        (
            "models:/health_models_openfda_serious_prediction_random_forest@champion",
            "health_models_openfda_serious_prediction_random_forest@champion",
            "4",
        ),
        (
            "models:/health_models_openfda_serious_prediction_random_forest/4",
            "health_models_openfda_serious_prediction_random_forest@4",
            "4",
        ),
    ],
)
def test_loads_registered_model_uri(
    tmp_path, monkeypatch, model_uri, model_key, expected_version
):
    model = object()
    loaded_uris = []
    monkeypatch.setenv("MLFLOW_MODEL_URI", model_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    def get_model_version_by_alias(self, name, alias):
        assert name == "health_models_openfda_serious_prediction_random_forest"
        assert alias == "champion"
        return SimpleNamespace(version="4")

    monkeypatch.setattr(
        model_server.MlflowClient,
        "get_model_version_by_alias",
        get_model_version_by_alias,
    )
    monkeypatch.setattr(
        model_server.ModelServer,
        "_load_mlflow_model",
        lambda self, uri: loaded_uris.append(uri) or model,
    )

    server = model_server.ModelServer(models_dir=str(tmp_path))

    assert server.models[model_key] is model
    assert server.model_info[model_key]["model_version"] == expected_version
    assert loaded_uris == [f"models:/{model_key.split('@')[0]}/{expected_version}"]