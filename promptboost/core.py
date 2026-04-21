"""Compatibility facade for the PromptBoost core pipeline."""

from __future__ import annotations

from typing import Any

from promptboost.adapters import select_adapter
from promptboost.db import save_boost_run
from promptboost.pipeline import boost as pipeline_boost
from promptboost.pipeline import classify_task


def boost(raw_prompt: str, target_harness: str = "codex", target_model: str = "gpt", **_: Any):
    adapter = select_adapter(target_harness, target_model)
    result = pipeline_boost(
        raw_prompt,
        adapter["normalized_harness"],
        adapter["normalized_model"],
    )
    saved = save_boost_run(result)
    data = result.model_dump()
    data["run_id"] = saved.id
    return data


boost_prompt = boost
classify_prompt = classify_task
classify = classify_task
classify_raw_prompt = classify_task
