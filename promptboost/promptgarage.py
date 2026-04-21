"""Read-only PromptGarage access."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import promptgarage_db_path
from .schemas import PromptGarageSample


def _readonly_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?immutable=1"


def connect_promptgarage(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or promptgarage_db_path()
    conn = sqlite3.connect(_readonly_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def promptgarage_available(path: Path | None = None) -> bool:
    return (path or promptgarage_db_path()).exists()


def sample_prompts(limit: int = 8, query: str | None = None, path: Path | None = None) -> list[PromptGarageSample]:
    db_path = path or promptgarage_db_path()
    if not db_path.exists():
        return []

    limit = max(1, min(int(limit), 50))
    with connect_promptgarage(db_path) as conn:
        if query:
            pattern = f"%{query.strip()}%"
            rows = conn.execute(
                """
                SELECT id, title, content, source, tags, model, rating, is_starred, updated_at
                FROM prompts
                WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                ORDER BY is_starred DESC, rating DESC, datetime(updated_at) DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, title, content, source, tags, model, rating, is_starred, updated_at
                FROM prompts
                ORDER BY is_starred DESC, rating DESC, datetime(updated_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [
        PromptGarageSample(
            id=int(row["id"]),
            title=row["title"],
            content=_trim(row["content"], 420),
            source=row["source"],
            tags=row["tags"],
            model=row["model"],
            rating=row["rating"],
            is_starred=bool(row["is_starred"]),
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def _trim(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."
