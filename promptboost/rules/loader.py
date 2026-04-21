"""Rule-pack loading for PromptBoost."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from promptboost.config import RULES_DIR


class RuleNotFoundError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def load_harness(harness_id: str) -> dict[str, Any]:
    path = RULES_DIR / "harnesses" / f"{harness_id}.json"
    if not path.exists():
        raise RuleNotFoundError(f"Unknown harness: {harness_id}")
    return _load_json(path)


@lru_cache(maxsize=None)
def load_model(model_id: str) -> dict[str, Any]:
    path = RULES_DIR / "models" / f"{model_id}.json"
    if not path.exists():
        raise RuleNotFoundError(f"Unknown model: {model_id}")
    return _load_json(path)


@lru_cache(maxsize=1)
def load_task_types() -> dict[str, Any]:
    return _load_json(RULES_DIR / "task_types.json")


def available_harnesses() -> list[str]:
    return sorted(
        path.stem for path in (RULES_DIR / "harnesses").glob("*.json") if path.stem != "hermes"
    )


def available_models() -> list[str]:
    return sorted(path.stem for path in (RULES_DIR / "models").glob("*.json"))
