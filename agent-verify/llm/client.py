"""One chat interface for three backends: local vLLM, OpenRouter, FreeLLMAPI.

All three speak the OpenAI chat-completions protocol, so the rest of the project
never needs to know which one it is talking to.

Experiment rule: always request a pinned, specific model ID. Router aliases such
as "auto" let the provider pick a different model per call, which silently breaks
fair comparisons and reproducibility, so they are rejected by default.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

try:  # optional: load keys from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BACKENDS = ("local", "openrouter", "freellmapi")

_DEFAULT_URLS = {
    "local": "http://localhost:11434/v1",  # Ollama default; vLLM uses :8000
    "openrouter": "https://openrouter.ai/api/v1",
    "freellmapi": "http://localhost:3001/v1",
}
_URL_ENV = {
    "local": "LOCAL_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
    "freellmapi": "FREELLMAPI_BASE_URL",
}
_KEY_ENV = {
    "local": "LOCAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "freellmapi": "FREELLMAPI_KEY",
}

# HTTP statuses that will not be fixed by retrying.
_NON_RETRYABLE = {400, 401, 403, 404, 422}


class LLMError(RuntimeError):
    """Raised when a call fails after all retries."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    served_model: str  # the model the server says actually answered
    latency_s: float


def is_router_alias(model: str) -> bool:
    """True for 'auto', 'auto-smart', 'freellmapi/auto', 'openrouter/auto', ..."""
    last = model.strip().lower().split("/")[-1]
    return last == "auto" or last.startswith("auto-")


def get_client(backend: str) -> OpenAI:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
    base_url = os.environ.get(_URL_ENV[backend]) or _DEFAULT_URLS[backend]
    api_key = os.environ.get(_KEY_ENV[backend])
    if not api_key:
        if backend == "local":
            api_key = "none"  # local vLLM does not check keys
        else:
            raise RuntimeError(
                f"Missing {_KEY_ENV[backend]}. Add it to your .env file "
                f"(see .env.example) or export it in your shell."
            )
    # max_retries=0: we do our own retry/backoff so every attempt is visible.
    return OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
    retries: int = 4,
    allow_router_alias: bool = False,
) -> LLMResponse:
    """Send one chat request with exponential backoff. Returns an LLMResponse."""
    if is_router_alias(model) and not allow_router_alias:
        raise ValueError(
            f"Model {model!r} is a router alias. Use a pinned model ID for experiments "
            f"(pass allow_router_alias=True only for quick manual tests)."
        )

    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            start = time.time()
            r = client.chat.completions.create(**kwargs)
            latency = time.time() - start
            usage = getattr(r, "usage", None)
            prompt_t = getattr(usage, "prompt_tokens", 0) or 0
            comp_t = getattr(usage, "completion_tokens", 0) or 0
            total_t = getattr(usage, "total_tokens", 0) or (prompt_t + comp_t)
            text = r.choices[0].message.content or ""
            return LLMResponse(
                text=text,
                prompt_tokens=prompt_t,
                completion_tokens=comp_t,
                total_tokens=total_t,
                served_model=getattr(r, "model", "") or "",
                latency_s=latency,
            )
        except Exception as e:  # noqa: BLE001 - re-raised as LLMError below
            last_error = e
            status = getattr(e, "status_code", None)
            if status in _NON_RETRYABLE:
                break
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt * 2, 30))  # 2s, 4s, 8s ... capped at 30s
    raise LLMError(f"LLM call failed after retries: {last_error!r}")
