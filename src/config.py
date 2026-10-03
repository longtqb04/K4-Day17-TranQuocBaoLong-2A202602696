from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig

    model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.0,
        api_key=None,
        base_url=None
    )

    judge_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.0,
        api_key=None,
        base_url=None
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # TODO: read env vars for one of the supported providers.
    # Example knobs:
    # - LLM_PROVIDER / LLM_MODEL
    # - OPENAI_API_KEY
    # - GEMINI_API_KEY
    # - ANTHROPIC_API_KEY
    # - OLLAMA_BASE_URL
    # - OPENROUTER_API_KEY
    # - CUSTOM_BASE_URL / CUSTOM_API_KEY
    # TODO: create `root / "state"`.
    # TODO: choose sensible defaults for compact memory.

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=root / "state",
        compact_threshold_tokens=1000,
        compact_keep_messages=10,
        model=ProviderConfig(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.0,
            api_key=None,
            base_url=None
        ),
        judge_model=ProviderConfig(
            provider="openai",
            model_name="gpt-4o",
            temperature=0.0,
            api_key=None,
            base_url=None
        )
    )
