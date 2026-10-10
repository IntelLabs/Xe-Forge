import os
from types import SimpleNamespace

import pytest

dspy = pytest.importorskip("dspy")
litellm = pytest.importorskip("litellm")

from xe_forge.llm_setup import configure_dspy, is_bedrock  # noqa: E402

BEDROCK = "bedrock/us.anthropic.claude-sonnet-5-5"


def _llm(model, api_base=None, api_key=None):
    return SimpleNamespace(
        model=model, api_base=api_base, api_key=api_key, temperature=1, max_tokens=16
    )


@pytest.fixture
def configured(monkeypatch):
    """configure_dspy with its globals restored afterwards; returns the LM it registered."""
    for var in ("OPENAI_API_BASE", "OPENAI_API_KEY", "AWS_REGION_NAME", "AWS_REGION"):
        monkeypatch.delenv(var, raising=False)
    for attr in ("client_session", "aclient_session", "ssl_verify"):
        monkeypatch.setattr(litellm, attr, getattr(litellm, attr, None), raising=False)
    got = {}
    monkeypatch.setattr(dspy, "configure", lambda **kw: got.update(kw))

    def run(llm):
        configure_dspy(llm)
        return got["lm"]

    return run


def test_is_bedrock():
    assert is_bedrock(BEDROCK)
    assert not is_bedrock("openai/gpt-4o")
    assert not is_bedrock("gpt-bedrock")


def test_bedrock_uses_chat_and_no_openai_endpoint(configured):
    lm = configured(_llm(BEDROCK, api_base="https://proxy", api_key="k"))
    assert (lm.model, lm.model_type) == (BEDROCK, "chat")
    assert "api_base" not in lm.kwargs and "api_key" not in lm.kwargs
    assert "OPENAI_API_BASE" not in os.environ and "OPENAI_API_KEY" not in os.environ


@pytest.mark.parametrize(
    "env, region",
    [
        ({}, "us-east-1"),
        ({"AWS_REGION": "eu-west-1"}, "eu-west-1"),
        ({"AWS_REGION": "eu-west-1", "AWS_REGION_NAME": "us-west-2"}, "us-west-2"),
    ],
)
def test_bedrock_region(configured, monkeypatch, env, region):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert configured(_llm(BEDROCK)).kwargs["aws_region_name"] == region


def test_openai_path_unchanged(configured):
    lm = configured(_llm("openai/gpt-4o", api_base="https://proxy", api_key="k"))
    assert lm.model_type == "responses"
    assert lm.kwargs["api_base"] == "https://proxy"
    assert os.environ["OPENAI_API_BASE"] == "https://proxy"


@pytest.mark.parametrize("model", [BEDROCK, "openai/gpt-4o"])
def test_tls_verified(configured, model):
    configured(_llm(model))
    assert litellm.ssl_verify is True


def test_bedrock_request_routes_offline(configured, monkeypatch):
    # mock_response stops litellm before the network, so no AWS credentials are needed.
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        monkeypatch.delenv(var, raising=False)
    lm = configured(_llm(BEDROCK))
    assert litellm.get_llm_provider(lm.model)[1] == "bedrock"
    assert lm("ping", mock_response="pong") == ["pong"]
