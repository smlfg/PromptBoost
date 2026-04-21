from __future__ import annotations

from conftest import (
    call_with_supported_kwargs,
    install_fake_adapter,
    maybe_import,
    resolve_callable,
    sqlite_contains_text,
    sqlite_user_tables,
    text_blob,
    unload_promptboost_modules,
)


BOOST_MODULES = (
    "promptboost",
    "promptboost.core",
    "promptboost.boost",
)

BOOST_FUNCS = (
    "boost",
    "boost_prompt",
)


def _patch_guards_to_pass(monkeypatch):
    for module_name in (
        "promptboost.guards",
        "promptboost.guardrails",
        "promptboost.rules.guards",
        "promptboost.rules",
        "promptboost.core",
        "promptboost.boost",
    ):
        module = maybe_import(module_name)
        if module is None:
            continue
        for attr_name in (
            "ensure_no_new_facts",
            "check_no_new_facts",
            "validate_no_new_facts",
            "guard_no_new_facts",
        ):
            monkeypatch.setattr(module, attr_name, lambda *_args, **_kwargs: True, raising=False)


def _call_boost(raw_prompt: str, target_harness: str, target_model: str):
    boost = resolve_callable(BOOST_MODULES, BOOST_FUNCS)
    return call_with_supported_kwargs(
        boost,
        raw_prompt=raw_prompt,
        prompt=raw_prompt,
        text=raw_prompt,
        target_harness=target_harness,
        harness=target_harness,
        target_model=target_model,
        model=target_model,
    )


def test_boost_persists_run_to_local_sqlite(promptboost_db_path, monkeypatch):
    unload_promptboost_modules()
    raw_prompt = (
        "Create a PromptBoost test prompt about FastAPI, local SQLite, and no paid API calls."
    )
    boosted_prompt = (
        "Create a focused PromptBoost test prompt about FastAPI, local SQLite, "
        "and no paid API calls."
    )

    resolve_callable(BOOST_MODULES, BOOST_FUNCS)
    install_fake_adapter(monkeypatch, boosted_prompt)
    _patch_guards_to_pass(monkeypatch)

    result = _call_boost(raw_prompt, target_harness="codex", target_model="test-model")

    assert boosted_prompt.lower() in text_blob(result)
    assert promptboost_db_path.exists()
    assert sqlite_user_tables(promptboost_db_path)
    assert sqlite_contains_text(promptboost_db_path, raw_prompt)
    assert sqlite_contains_text(promptboost_db_path, boosted_prompt)
    assert sqlite_contains_text(promptboost_db_path, "codex")
    assert sqlite_contains_text(promptboost_db_path, "test-model")
