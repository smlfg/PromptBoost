"""Local SQLite persistence for PromptBoost boost runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import promptboost_db_path
from .schemas import BoostResult, BoostRun


SCHEMA = """
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
    warnings_json TEXT NOT NULL,
    result_json TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or promptboost_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(boost_runs)").fetchall()
        }
        if "result_json" not in columns:
            conn.execute("ALTER TABLE boost_runs ADD COLUMN result_json TEXT")


def save_boost_run(result: BoostResult, db_path: Path | None = None) -> BoostRun:
    init_db(db_path)
    with connect(db_path) as conn:
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
                warnings_json,
                result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.raw_prompt,
                result.detected_task_type,
                result.target_harness,
                result.target_model,
                json.dumps(result.adapter_rules, sort_keys=True),
                result.boosted_prompt,
                int(result.no_new_facts_passed),
                json.dumps(result.warnings),
                result.model_dump_json(),
            ),
        )
        row = conn.execute("SELECT * FROM boost_runs WHERE id = ?", (cur.lastrowid,)).fetchone()
    return row_to_run(row)


def list_boost_runs(limit: int = 20, db_path: Path | None = None) -> list[BoostRun]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM boost_runs ORDER BY datetime(created_at) DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_run(row) for row in rows]


def row_to_run(row: sqlite3.Row | dict[str, Any]) -> BoostRun:
    result_json = row["result_json"] if "result_json" in row.keys() else None
    result = json.loads(result_json) if result_json else {}
    return BoostRun(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        raw_prompt=str(row["raw_prompt"]),
        detected_task_type=str(row["detected_task_type"]),
        target_harness=str(row["target_harness"]),
        target_model=str(row["target_model"]),
        adapter_rules=json.loads(row["adapter_rules_json"]),
        boosted_prompt=str(row["boosted_prompt"]),
        no_new_facts_passed=bool(row["no_new_facts_passed"]),
        warnings=json.loads(row["warnings_json"]),
        result=result,
        score=result.get("score") if result else None,
    )
