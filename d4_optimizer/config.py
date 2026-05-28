"""
System configuration, environment variable resolution, and API key management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
AFFIXES_DB_PATH = DATA_DIR / "affixes.json"
UNIQUES_DB_PATH = DATA_DIR / "uniques.json"


class LLMProvider(str, Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"


class ItemTier(str, Enum):
    NORMAL = "normal"
    MAGIC = "magic"
    RARE = "rare"
    SACRED = "sacred"
    ANCESTRAL = "ancestral"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.CLAUDE
    model: str = "claude-opus-4-8"
    temperature: float = 0.1
    max_output_tokens: int = 8192
    # Gemini alternative
    gemini_model: str = "gemini-2.5-pro"

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"Temperature must be in [0.0, 1.0], got {self.temperature}")


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Maximum number of stash items to include in a single LLM call.
    # Items beyond this are batched across multiple calls.
    stash_batch_size: int = 50
    # Item power threshold below which stash items are pre-filtered as non-candidates.
    min_item_power: int = 700
    debug: bool = False


def load_config() -> AppConfig:
    """Build AppConfig from environment variables, falling back to defaults."""
    provider_str = os.getenv("D4_LLM_PROVIDER", LLMProvider.CLAUDE.value).lower()
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        raise ValueError(
            f"Unsupported LLM provider '{provider_str}'. "
            f"Valid options: {[p.value for p in LLMProvider]}"
        )

    model_default = "claude-opus-4-8" if provider == LLMProvider.CLAUDE else "gemini-2.5-pro"
    model = os.getenv("D4_LLM_MODEL", model_default)

    temperature_raw = os.getenv("D4_LLM_TEMPERATURE", "0.1")
    try:
        temperature = float(temperature_raw)
    except ValueError:
        raise ValueError(f"D4_LLM_TEMPERATURE must be a float, got '{temperature_raw}'")

    return AppConfig(
        llm=LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=int(os.getenv("D4_LLM_MAX_TOKENS", "8192")),
        ),
        stash_batch_size=int(os.getenv("D4_STASH_BATCH_SIZE", "50")),
        min_item_power=int(os.getenv("D4_MIN_ITEM_POWER", "700")),
        debug=os.getenv("D4_DEBUG", "").lower() in ("1", "true", "yes"),
    )


def get_api_key(provider: LLMProvider) -> str:
    """Retrieve the API key for the specified provider from environment variables."""
    env_map: dict[LLMProvider, str] = {
        LLMProvider.CLAUDE: "ANTHROPIC_API_KEY",
        LLMProvider.GEMINI: "GEMINI_API_KEY",
    }
    env_var = env_map[provider]
    key = os.getenv(env_var, "")
    if not key:
        raise EnvironmentError(
            f"API key not found. Set the '{env_var}' environment variable."
        )
    return key
