from __future__ import annotations

import pytest

from conftest import call_with_supported_kwargs, resolve_callable, text_blob


ADAPTER_MODULES = (
    "promptboost.adapters",
    "promptboost.adapter",
    "promptboost.core",
)

ADAPTER_SELECTOR_FUNCS = (
    "select_adapter",
    "get_adapter",
    "resolve_adapter",
)


def _select_adapter(target_harness: str, target_model: str):
    selector = resolve_callable(ADAPTER_MODULES, ADAPTER_SELECTOR_FUNCS)
    return call_with_supported_kwargs(
        selector,
        target_harness=target_harness,
        harness=target_harness,
        target_model=target_model,
        model=target_model,
    )


def test_select_adapter_uses_harness_and_model():
    adapter = _select_adapter("codex", "gpt-5.4")

    blob = text_blob(adapter)
    assert "codex" in blob
    assert any(token in blob for token in ("gpt", "5.4", "openai"))


def test_select_adapter_distinguishes_supported_targets():
    codex = _select_adapter("codex", "gpt-5.4")
    opencode = _select_adapter("opencode", "minimax-m2.7")

    assert text_blob(codex) != text_blob(opencode)
    assert "opencode" in text_blob(opencode)
    assert any(token in text_blob(opencode) for token in ("minimax", "m2.7", "m2"))


def test_select_adapter_rejects_unknown_targets():
    with pytest.raises((KeyError, LookupError, ValueError, NotImplementedError)):
        _select_adapter("unknown-harness", "unknown-model")
