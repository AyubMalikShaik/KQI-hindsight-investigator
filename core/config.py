"""Configuration loaded from environment / .env.

No secret is ever read from a file other than .env, and nothing here is logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

ENV_FILE = PROJECT_ROOT / ".env"


def _load_dotenv() -> None:
    """Minimal .env loader. Existing environment variables win.

    Reads as utf-8-sig so a BOM written by a Windows editor does not corrupt the
    first key name.
    """
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model_primary: str
    model_fallback: str
    timeout_s: float
    max_retries: int

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("gsk_REPLACE")


@dataclass(frozen=True)
class HindsightConfig:
    base_url: str
    api_key: str
    bank: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url) and bool(self.api_key)


@dataclass(frozen=True)
class BudgetConfig:
    """Hard caps that keep every investigation bounded and traceable."""

    max_tool_calls: int
    max_validation_loops: int
    max_llm_turns: int
    max_tokens_total: int
    tool_timeout_s: float
    confidence_threshold: float
    max_repeat_calls: int
    recall_budget: str
    recall_max_tokens: int

    @classmethod
    def defaults(cls) -> "BudgetConfig":
        return cls(
            max_tool_calls=15,
            max_validation_loops=2,
            max_llm_turns=10,
            max_tokens_total=120_000,
            tool_timeout_s=20.0,
            confidence_threshold=0.75,
            max_repeat_calls=3,
            recall_budget="mid",
            recall_max_tokens=2048,
        )


@dataclass(frozen=True)
class Settings:
    tenant: str
    groq: GroqConfig
    hindsight: HindsightConfig
    budget: BudgetConfig = field(default_factory=BudgetConfig.defaults)
    db_path: Path = ARTIFACTS_DIR / "lumen.duckdb"

    @property
    def bank(self) -> str:
        """One Hindsight bank per tenant, so memories never leak across orgs."""
        return f"{self.hindsight.bank}::{self.tenant}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    override = _get("LUMEN_DB_PATH")
    return Settings(
        tenant=_get("INVESTIGATOR_TENANT", "acme"),
        groq=GroqConfig(
            api_key=_get("GROQ_API_KEY"),
            model_primary=_get("GROQ_MODEL_PRIMARY", "openai/gpt-oss-120b"),
            model_fallback=_get("GROQ_MODEL_FALLBACK", "openai/gpt-oss-20b"),
            timeout_s=_get_float("GROQ_TIMEOUT_S", 90.0),
            max_retries=_get_int("GROQ_MAX_RETRIES", 3),
        ),
        hindsight=HindsightConfig(
            base_url=_get("HINDSIGHT_BASE_URL"),
            api_key=_get("HINDSIGHT_API_KEY"),
            bank=_get("HINDSIGHT_BANK", "anomaly-investigator"),
        ),
        budget=BudgetConfig.defaults(),
        db_path=Path(override) if override else ARTIFACTS_DIR / "lumen.duckdb",
    )
