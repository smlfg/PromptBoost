from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from promptboost.config import DEFAULT_PROMPTGARAGE_DB
from promptboost.promptgarage import connect_promptgarage, sample_prompts


def test_promptgarage_readonly_connection_if_available() -> None:
    if not DEFAULT_PROMPTGARAGE_DB.exists():
        pytest.skip("PromptGarage DB is not available")

    with connect_promptgarage(DEFAULT_PROMPTGARAGE_DB) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        assert count >= 0
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE promptboost_write_probe (id INTEGER)")


def test_promptgarage_sampling_from_temp_db(tmp_path: Path) -> None:
    db_path = tmp_path / "prompts.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE prompts (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            is_starred INTEGER DEFAULT 0,
            tags TEXT,
            model TEXT,
            rating INTEGER,
            use_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "INSERT INTO prompts (title, content, tags, model, rating, is_starred) VALUES (?, ?, ?, ?, ?, ?)",
        ("Sample", "A useful prompt example", "test", "gpt", 5, 1),
    )
    conn.commit()
    conn.close()

    samples = sample_prompts(path=db_path)

    assert len(samples) == 1
    assert samples[0].title == "Sample"
