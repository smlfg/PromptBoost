from __future__ import annotations

import pytest

from conftest import as_mapping, call_with_supported_kwargs, resolve_callable, text_blob


GUARD_MODULES = (
    "promptboost.guards",
    "promptboost.guardrails",
    "promptboost.rules.guards",
    "promptboost.rules",
    "promptboost.core",
)

GUARD_FUNCS = (
    "ensure_no_new_facts",
    "check_no_new_facts",
    "validate_no_new_facts",
    "guard_no_new_facts",
)


def _call_guard(original: str, candidate: str):
    guard = resolve_callable(GUARD_MODULES, GUARD_FUNCS)
    return call_with_supported_kwargs(
        guard,
        original=original,
        source=original,
        source_text=original,
        original_prompt=original,
        raw_prompt=original,
        before=original,
        candidate=candidate,
        output=candidate,
        rewritten_prompt=candidate,
        boosted_prompt=candidate,
        after=candidate,
    )


def _guard_allows(result) -> bool:
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    if isinstance(result, (list, tuple, set)):
        return len(result) == 0

    mapping = as_mapping(result)
    if mapping:
        for key in ("ok", "allowed", "passed", "valid", "is_valid"):
            if key in mapping:
                return bool(mapping[key])
        for key in ("has_new_facts", "new_facts_detected"):
            if key in mapping:
                return not bool(mapping[key])
        for key in ("violations", "new_facts", "errors"):
            if key in mapping:
                return not bool(mapping[key])

    return bool(result)


def test_no_new_facts_guard_allows_grounded_rewrite():
    original = (
        "PromptBoost v0.1 has a FastAPI app, local SQLite storage, and a "
        "boost(raw_prompt, target_harness, target_model) entry point."
    )
    candidate = (
        "Use FastAPI, persist runs in local SQLite, and expose "
        "boost(raw_prompt, target_harness, target_model) for PromptBoost v0.1."
    )

    result = _call_guard(original, candidate)

    assert _guard_allows(result), text_blob(result)


def test_no_new_facts_guard_rejects_invented_capabilities():
    original = (
        "PromptBoost v0.1 has a FastAPI app, local SQLite storage, and a "
        "boost(raw_prompt, target_harness, target_model) entry point."
    )
    candidate = (
        "PromptBoost v0.1 adds FastAPI, local SQLite, Redis queues, OAuth login, "
        "and Pinecone vector search."
    )

    try:
        result = _call_guard(original, candidate)
    except (AssertionError, LookupError, ValueError) as exc:
        assert any(term in str(exc).lower() for term in ("redis", "oauth", "pinecone", "new fact"))
        return

    assert not _guard_allows(result), text_blob(result)
    assert any(term in text_blob(result) for term in ("redis", "oauth", "pinecone", "new fact"))


def test_no_new_facts_guard_allows_harmless_meta_terms():
    original = "Summarize the provided PDF quellentreu in Markdown."
    candidate = "Fully summarize the provided PDF quellentreu in Markdown with a complete verification note."

    result = _call_guard(original, candidate)

    assert _guard_allows(result), text_blob(result)
    mapping = as_mapping(result)
    assert "fully" in mapping.get("harmless_meta_terms", [])
