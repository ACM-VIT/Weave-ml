"""Real artifact/device configuration checks; no external service required."""
from pathlib import Path

import pytest

from inference.ranker_service import RankerService
from retrieval.v2_retriever import QdrantV2Retriever

ARTIFACTS = Path(__file__).resolve().parents[1] / "inference"


def load_ranker(**kwargs):
    return RankerService(model_path=str(ARTIFACTS / "heavy_ranker.pt"),
                         scaler_path=str(ARTIFACTS / "feature_scaler.json"),
                         manifest_path=str(ARTIFACTS / "model_manifest.json"),
                         require_production_manifest=True, **kwargs)


def test_explicit_cpu_ignores_automatic_cuda_preference(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    service = load_ranker(device="cpu")
    assert service.ready and service.production_qualified
    assert service.device.type == "cpu"
    assert next(service.model.parameters()).device.type == "cpu"
    assert service.qualification_overridden  # Preserve the existing manifest provenance.


def test_factory_honors_explicit_cpu_on_cuda_capable_host(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("V2_HEAVY_RANKER_DEVICE", "cpu")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    retriever = QdrantV2Retriever(client=object(), heavy_ranker_enabled=True,
                                  heavy_ranker_required=True, allow_unqualified_ranker=False,
                                  minimum_eligible_repositories=0)
    assert retriever.ranker is not None
    assert retriever.ranker.device.type == "cpu"
    assert retriever.heavy_ranker_required


def test_explicit_unavailable_cuda_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        load_ranker(device="cuda")


def test_invalid_runtime_device_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("V2_HEAVY_RANKER_DEVICE", "cdua")
    with pytest.raises(ValueError, match="V2_HEAVY_RANKER_DEVICE"):
        QdrantV2Retriever(client=object(), heavy_ranker_enabled=False,
                          minimum_eligible_repositories=0)
