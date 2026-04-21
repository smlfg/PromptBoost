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
