"""
StudyMind — LLMClient: one place to call any model through LiteLLM.

The model is one config line (LLM_MODEL), e.g. ollama/qwen2.5:7b,
ollama/gemma3:4b, groq/..., anthropic/.... Every call logs tokens, cost and
latency; Langfuse tracing turns on when its keys are set.
"""

import logging
import os
import time

# Use LiteLLM's bundled price map instead of fetching it from GitHub on import
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

from core.config import settings

log = logging.getLogger("studymind.llm")


def setup_tracing() -> bool:
    """Send every LiteLLM call to Langfuse if keys are configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    # LiteLLM's OTEL Langfuse callback reads these env vars; the legacy
    # "langfuse" callback only supports old Langfuse SDKs.
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    if "langfuse_otel" not in litellm.callbacks:
        litellm.callbacks.append("langfuse_otel")
    return True


class LLMClient:
    async def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Single-turn completion; returns the text. Raises on provider errors."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        model = kwargs.pop("model", settings.llm_model)
        if model.startswith("ollama/"):
            kwargs.setdefault("api_base", settings.ollama_base_url)

        start = time.perf_counter()
        resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
        latency = time.perf_counter() - start

        usage = getattr(resp, "usage", None)
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0  # unknown pricing (local models)
        log.info(
            "llm model=%s prompt_tokens=%s completion_tokens=%s cost_usd=%.6f latency_s=%.2f",
            model,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            cost,
            latency,
        )
        return resp.choices[0].message.content or ""


llm = LLMClient()
