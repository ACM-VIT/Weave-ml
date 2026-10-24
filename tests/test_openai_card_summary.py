"""Direct OpenAI requests preserve the existing bounded summary contract."""
import json
import pytest

from summarization.provider import OpenRouterSummaryProvider
from summarization.settings import SummarySettings
from tests.test_production_config import valid_production_env
from scripts.validate_production_config import validate_production_config


def test_openai_card_summary_uses_official_endpoint_and_supported_budget():
    calls = []

    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"message": {"content": '{"summary":"A focused repository discovery tool."}'}}]}

    class Session:
        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    settings = SummarySettings(provider="openai", api_url="https://api.openai.com/v1/chat/completions",
                               model_id="gpt-6-luna", api_key="test-openai-only-key")
    provider = OpenRouterSummaryProvider(settings, session=Session())
    assert json.loads(provider.generate("Repository: weave/discovery"))["summary"] == "A focused repository discovery tool."
    url, request = calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-openai-only-key"
    payload = request["json"]
    assert payload["model"] == "gpt-6-luna"
    assert payload["max_completion_tokens"] == 240
    assert payload["reasoning_effort"] == "none"
    assert "max_tokens" not in payload
    assert "provider" not in payload
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_openai_provider_defaults_to_its_own_endpoint(monkeypatch):
    monkeypatch.setenv("SUMMARY_PROVIDER", "openai")
    monkeypatch.delenv("SUMMARY_API_URL", raising=False)
    monkeypatch.delenv("SUMMARY_MODEL_ID", raising=False)
    settings = SummarySettings.from_env()
    assert settings.api_url == "https://api.openai.com/v1/chat/completions"
    assert settings.model_id == "gpt-6-luna"


@pytest.mark.parametrize("url", ["https://openrouter.ai/api/v1/chat/completions", "https://other.example/v1/chat/completions"])
def test_openai_provider_rejects_cross_provider_key_destination(url):
    with pytest.raises(ValueError, match="OpenAI endpoint"):
        SummarySettings(provider="openai", api_url=url, model_id="gpt-6-luna")


def test_production_preflight_accepts_explicit_openai_configuration():
    env = valid_production_env()
    env.update(SUMMARY_PROVIDER="openai", SUMMARY_API_URL="https://api.openai.com/v1/chat/completions", SUMMARY_MODEL_ID="gpt-6-luna")
    assert validate_production_config(env) == []


def test_production_preflight_rejects_openai_key_routed_to_openrouter():
    env = valid_production_env()
    env.update(SUMMARY_PROVIDER="openai", SUMMARY_API_URL="https://openrouter.ai/api/v1/chat/completions", SUMMARY_MODEL_ID="gpt-6-luna")
    assert any(issue.name == "SUMMARY_API_URL" for issue in validate_production_config(env))
