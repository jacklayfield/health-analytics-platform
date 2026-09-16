"""Tests for dataset-aware ML configuration."""

from types import SimpleNamespace

import pytest

from src.config.config_manager import ConfigManager, DatasetConfig


@pytest.fixture
def config_manager_with_datasets():
    manager = ConfigManager()
    manager._config = SimpleNamespace(
        datasets={
            "openfda": DatasetConfig(
                table="openfda_events",
                schema_version="1",
                entity_level="adverse_event",
                tasks=["serious_prediction"],
            )
        }
    )
    return manager


def test_get_dataset_config(config_manager_with_datasets):
    dataset = config_manager_with_datasets.get_dataset_config("openfda")

    assert dataset.table == "openfda_events"
    assert dataset.entity_level == "adverse_event"


def test_validate_dataset_task_rejects_unsupported_task(
    config_manager_with_datasets,
):
    with pytest.raises(ValueError, match="not configured for dataset"):
        config_manager_with_datasets.validate_dataset_task(
            "openfda", "condition_prediction"
        )


def test_get_dataset_config_rejects_unknown_dataset(config_manager_with_datasets):
    with pytest.raises(KeyError, match="Dataset configuration not found"):
        config_manager_with_datasets.get_dataset_config("missing")
