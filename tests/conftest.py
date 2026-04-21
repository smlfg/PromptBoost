from __future__ import annotations

import importlib
import inspect
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "promptboost-test.db"
    monkeypatch.setenv("PROMPTBOOST_DB_PATH", str(db_path))
    return db_path


@pytest.fixture()
def promptboost_db_path(temp_db: Path) -> Path:
    return temp_db


def maybe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def import_any(*module_names: str):
    for module_name in module_names:
        module = maybe_import(module_name)
        if module is not None:
            return module
    raise ImportError(f"Could not import any module: {module_names}")


def resolve_callable(module_names: tuple[str, ...], func_names: tuple[str, ...]):
    for module_name in module_names:
        module = maybe_import(module_name)
        if module is None:
            continue
        for func_name in func_names:
            candidate = getattr(module, func_name, None)
            if callable(candidate):
                return candidate
    raise LookupError(f"Could not resolve callable {func_names} in {module_names}")


def call_with_supported_kwargs(func, **kwargs):
    sig = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return func(**{key: value for key, value in kwargs.items() if value is not None})

    supported = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters and value is not None
    }
    return func(**supported)


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def text_blob(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        return json.dumps(value, sort_keys=True, default=str).lower()
    except TypeError:
        return str(value).lower()


def unload_promptboost_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "promptboost" or module_name.startswith("promptboost."):
            sys.modules.pop(module_name, None)


def install_fake_adapter(monkeypatch: pytest.MonkeyPatch, boosted_prompt: str) -> None:
    def fake_boost(
        raw_prompt: str = "",
        prompt: str = "",
        text: str = "",
        target_harness: str = "codex",
        harness: str | None = None,
        target_model: str = "test-model",
        model: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        from promptboost.config import promptboost_db_path

        source = raw_prompt or prompt or text
        chosen_harness = harness or target_harness
        chosen_model = model or target_model
        db_path = promptboost_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS boost_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    raw_prompt TEXT NOT NULL,
                    detected_task_type TEXT NOT NULL,
                    target_harness TEXT NOT NULL,
                    target_model TEXT NOT NULL,
                    adapter_rules_json TEXT NOT NULL,
                    boosted_prompt TEXT NOT NULL,
                    no_new_facts_passed INTEGER NOT NULL,
                    warnings_json TEXT NOT NULL
                )
                """
            )
            cur = conn.execute(
                """
                INSERT INTO boost_runs (
                    raw_prompt,
                    detected_task_type,
                    target_harness,
                    target_model,
                    adapter_rules_json,
                    boosted_prompt,
                    no_new_facts_passed,
                    warnings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    "general_agent_task",
                    chosen_harness,
                    chosen_model,
                    "{}",
                    boosted_prompt,
                    1,
                    "[]",
                ),
            )
        return {
            "run_id": cur.lastrowid,
            "raw_prompt": source,
            "boosted_prompt": boosted_prompt,
            "target_harness": chosen_harness,
            "target_model": chosen_model,
        }

    for module_name in ("promptboost", "promptboost.core", "promptboost.boost"):
        module = maybe_import(module_name)
        if module is not None:
            monkeypatch.setattr(module, "boost", fake_boost, raising=False)
            monkeypatch.setattr(module, "boost_prompt", fake_boost, raising=False)


def sqlite_user_tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return [row[0] for row in rows]


def sqlite_contains_text(path: Path, needle: str) -> bool:
    with sqlite3.connect(path) as conn:
        for table in sqlite_user_tables(path):
            columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
            text_cols = [col[1] for col in columns if "TEXT" in col[2].upper()]
            for col in text_cols:
                rows = conn.execute(f"SELECT 1 FROM {table} WHERE {col} LIKE ? LIMIT 1", (f"%{needle}%",)).fetchall()
                if rows:
                    return True
    return False
