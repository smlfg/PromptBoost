"""Configuration helpers for local PromptBoost paths."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
RULES_DIR = PACKAGE_DIR / "rules"
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

DEFAULT_PROMPTGARAGE_DB = Path("/home/smlflg/Projekte/PromptGarage/prompts.db")


def promptboost_db_path() -> Path:
    return Path(os.environ.get("PROMPTBOOST_DB_PATH", PROJECT_DIR / "promptboost.db"))


def promptgarage_db_path() -> Path:
    return Path(os.environ.get("PROMPTGARAGE_DB_PATH", DEFAULT_PROMPTGARAGE_DB))


def llm_base_url() -> str:
    return os.environ.get("PROMPTBOOST_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def llm_api_key() -> str | None:
    return os.environ.get("PROMPTBOOST_LLM_API_KEY")


def llm_model() -> str:
    return os.environ.get("PROMPTBOOST_LLM_MODEL", "gpt-5.4-mini")


def llm_timeout_seconds() -> float:
    raw = os.environ.get("PROMPTBOOST_LLM_TIMEOUT_SECONDS", "20")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 20.0
