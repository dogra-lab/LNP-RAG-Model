"""LLM provider selection and the OpenAI client's response handling (no network)."""
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from lnp_rag.config import RetrievalConfig
from lnp_rag.services import Services
from lnp_rag.synthesis.openai_llm import OpenAIClient
from lnp_rag.synthesis.prompts import SynthesisError


def _services(settings, **update):
    return Services(settings.model_copy(update={"synthesis_enabled": True, **update}), RetrievalConfig())


def test_provider_auto_selection(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert _services(settings).llm_provider is None
    assert _services(settings).llm is None
    svc = _services(settings, openai_api_key=SecretStr("sk-test"))
    assert svc.llm_provider == "openai"
    assert svc.llm.name == "openai" and svc.llm.model == "gpt-5.4-mini"
    assert _services(settings, openai_api_key=SecretStr("sk-test"), anthropic_api_key=SecretStr("a")).llm_provider == "anthropic"
    assert _services(settings, openai_api_key=SecretStr("sk-test"), anthropic_api_key=SecretStr("a"),
                     llm_provider="openai").llm_provider == "openai"
    assert _services(settings, llm_provider="anthropic").llm_provider is None


def _fake_completion(content, finish="stop", refusal=None):
    msg = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def _client(settings, response, captured):
    client = OpenAIClient(settings.model_copy(update={"openai_api_key": SecretStr("sk-test")}))

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


def test_openai_synthesis_uses_strict_schema(settings):
    captured = {}
    body = {"papers": [{"paper_key": "P1", "relevance": 80, "evidence_strength": "strong",
                        "study_type": "in vitro", "explanation": "x [P1-E1]",
                        "cited_evidence_ids": ["P1-E1"], "missing_evidence": ""}]}
    client = _client(settings, _fake_completion(json.dumps(body)), captured)
    out = client.synthesize("q", [{"key": "P1", "title": "T", "evidence": [
        {"id": "P1-E1", "type": "paragraph", "section": "Results", "page": 3, "text": "uptake <at> 4 h"}]}])
    assert out["P1"]["relevance"] == 80
    fmt = captured["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert "reasoning_effort" not in captured  # only sent when configured
    user = captured["messages"][1]["content"]
    assert 'id="P1-E1"' in user and "&lt;at&gt;" in user  # excerpts are escaped


@pytest.mark.parametrize("response, message", [
    (_fake_completion(None, refusal="no"), "declined"),
    (_fake_completion("{", finish="length"), "truncated"),
    (_fake_completion("not json"), "malformed"),
])
def test_openai_bad_responses_raise_synthesis_error(settings, response, message):
    client = _client(settings, response, {})
    with pytest.raises(SynthesisError, match=message):
        client.synthesize("q", [])
