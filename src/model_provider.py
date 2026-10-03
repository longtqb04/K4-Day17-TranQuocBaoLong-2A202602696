from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import os
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable provider settings, with credentials hidden from repr."""
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize aliases and reject unsupported providers."""
    name = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "anthorpic": "anthropic", "claude": "anthropic", "google": "gemini",
        "google_genai": "gemini", "google_generative_ai": "gemini",
        "open_router": "openrouter", "openai_compatible": "custom", "open_ai": "openai",
    }
    name = aliases.get(name, name)
    if name not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value!r}")
    return name


def build_chat_model(config: ProviderConfig) -> Any:
    """Lazily initialize the selected integration without making an API call.

    Offline mode requires no provider SDK. Explicit settings override env vars.
    """
    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        raise ValueError("model_name must not be empty")
    integrations = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "custom": ("langchain_openai", "ChatOpenAI"),
        "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    }
    key_vars = {
        "openai": ("OPENAI_API_KEY",), "custom": ("CUSTOM_API_KEY",),
        "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",), "openrouter": ("OPENROUTER_API_KEY",), "ollama": (),
    }
    api_key = config.api_key or next(
        (os.environ[name] for name in key_vars[provider] if os.environ.get(name)), None
    )
    url_vars = {
        "openai": "OPENAI_BASE_URL", "custom": "CUSTOM_BASE_URL",
        "gemini": "GEMINI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL",
        "ollama": "OLLAMA_BASE_URL", "openrouter": "OPENROUTER_BASE_URL",
    }
    base_url = config.base_url or os.environ.get(url_vars[provider])
    if provider == "custom" and not base_url:
        raise ValueError("custom requires base_url or CUSTOM_BASE_URL")
    if provider not in {"custom", "ollama"} and not api_key:
        raise ValueError(f"{provider} requires api_key or {' / '.join(key_vars[provider])}")
    kwargs: dict[str, Any] = {"model": config.model_name, "temperature": config.temperature}
    if provider == "custom":
        # Some local OpenAI-compatible servers do not require authentication.
        kwargs["api_key"] = api_key or "not-required"
    elif api_key and provider != "ollama":
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    module_name, class_name = integrations[provider]
    try:
        model_class = getattr(import_module(module_name), class_name)
    except ImportError as exc:
        package = module_name.replace("_", "-")
        raise ImportError(f"Install {package} to use the {provider} provider") from exc
    return model_class(**kwargs)
