from __future__ import annotations

from promptboost.db import list_boost_runs, save_boost_run
from promptboost.pipeline import boost


def test_boost_run_persistence(temp_db) -> None:
    result = boost("Fix a code bug and add tests.", "codex", "gpt")
    saved = save_boost_run(result, temp_db)
    runs = list_boost_runs(db_path=temp_db)

    assert saved.id == 1
    assert len(runs) == 1
    assert runs[0].raw_prompt == result.raw_prompt
    assert runs[0].adapter_rules["harness"]["id"] == "codex"
