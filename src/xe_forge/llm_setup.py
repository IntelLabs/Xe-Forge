"""Shared DSPy / LiteLLM configuration for all LLM-driven paths.

Supports OpenAI-compatible endpoints (the default) and AWS Bedrock. Bedrock is
selected automatically when the configured model id is prefixed with
``bedrock/`` (e.g. ``bedrock/us.anthropic.claude-sonnet-4-6``).

No credentials are read, logged, or stored here. For Bedrock, auth is handled
entirely by LiteLLM via the standard AWS environment variables
(``AWS_BEARER_TOKEN_BEDROCK`` or ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``),
and the region is resolved from ``AWS_REGION_NAME``/``AWS_REGION``.
"""

from __future__ import annotations

import os


def is_bedrock(model: str) -> bool:
    """Return True if ``model`` targets AWS Bedrock."""
    return model.startswith("bedrock/")


def configure_dspy(llm_config) -> None:
    """Build the DSPy LM from an LLM config section and register it globally.

    The OpenAI-compatible path is unchanged from the original inline setup.
    Bedrock uses the Converse (chat) API instead of the OpenAI Responses API
    and does not take an OpenAI ``api_base``; its region comes from the
    environment (default ``us-east-1``).
    """
    import dspy
    import httpx
    import litellm

    # TLS may terminate at a MITM proxy; keep verification disabled (as before)
    # and mirror it onto the async client used by Bedrock/Converse calls.
    litellm.ssl_verify = False
    litellm.client_session = httpx.Client(verify=False)
    litellm.aclient_session = httpx.AsyncClient(verify=False)

    if is_bedrock(llm_config.model):
        region = os.environ.get("AWS_REGION_NAME") or os.environ.get("AWS_REGION") or "us-east-1"
        lm = dspy.LM(
            model=llm_config.model,
            model_type="chat",
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            cache=False,
            aws_region_name=region,
        )
    else:
        if llm_config.api_base:
            os.environ["OPENAI_API_BASE"] = llm_config.api_base
        if llm_config.api_key:
            os.environ["OPENAI_API_KEY"] = llm_config.api_key
        lm = dspy.LM(
            model=llm_config.model,
            api_base=llm_config.api_base,
            model_type="responses",
            api_key=llm_config.api_key or "",
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            cache=False,
        )

    dspy.configure(lm=lm, warn_on_type_mismatch=False)
