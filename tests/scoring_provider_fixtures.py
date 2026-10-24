"""Shared production-artifact ranker fixture for retained integration tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from inference.ranker_service import RankerService


@pytest.fixture(scope="module")
def trained_ranker():
    folder = Path(inspect.getfile(RankerService)).parent
    return RankerService(
        model_path=str(folder / "heavy_ranker.pt"),
        scaler_path=str(folder / "feature_scaler.json"),
        manifest_path=str(folder / "model_manifest.json"),
        require_production_manifest=True,
        device="cpu",
    )
