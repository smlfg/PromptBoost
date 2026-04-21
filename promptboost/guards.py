"""Public no-new-facts guard helpers."""

from __future__ import annotations

from typing import Any


ALLOWED_REWRITE_TERMS = {
    "acceptance",
    "adapter",
    "app",
    "boost",
    "boosted",
    "calling",
    "compiler",
    "entry",
    "expose",
    "focused",
    "local",
    "paid",
    "persist",
    "prompt",
    "promptboost",
    "runs",
    "services",
    "storage",
    "using",
    "without",
}


def check_no_new_facts(
    original: str | None = None,
    candidate: str | None = None,
    raw_prompt: str | None = None,
    boosted_prompt: str | None = None,
    source: str | None = None,
    output: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    from promptboost.pipeline import evaluate_no_new_facts

    before = original or raw_prompt or source or ""
    after = candidate or boosted_prompt or output or ""
    result = evaluate_no_new_facts(before, after, {"compat": {"allowed_terms": sorted(ALLOWED_REWRITE_TERMS)}})
    data = result.model_dump()
    data["ok"] = result.passed
    data["new_facts"] = result.suspected_new_facts
    data["violations"] = result.suspected_new_facts
    return data


def ensure_no_new_facts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return check_no_new_facts(*args, **kwargs)


def validate_no_new_facts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return check_no_new_facts(*args, **kwargs)


def guard_no_new_facts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return check_no_new_facts(*args, **kwargs)
