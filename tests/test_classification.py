from __future__ import annotations

from conftest import as_mapping, call_with_supported_kwargs, resolve_callable, text_blob


CLASSIFIER_MODULES = (
    "promptboost.classifier",
    "promptboost.classification",
    "promptboost.rules.classifier",
    "promptboost.core",
)

CLASSIFIER_FUNCS = (
    "classify_prompt",
    "classify",
    "classify_task",
    "classify_raw_prompt",
)


def _classify(raw_prompt: str):
    classify = resolve_callable(CLASSIFIER_MODULES, CLASSIFIER_FUNCS)
    return call_with_supported_kwargs(
        classify,
        raw_prompt=raw_prompt,
        prompt=raw_prompt,
        text=raw_prompt,
        content=raw_prompt,
    )


def test_classifier_identifies_coding_debug_prompts():
    result = _classify(
        "Fix a failing pytest suite in a FastAPI project and keep the public API stable."
    )

    blob = text_blob(result)
    assert any(
        label in blob
        for label in ("coding", "code", "software", "debug", "test", "pytest")
    )

    mapping = as_mapping(result)
    if mapping:
        assert any(
            key in mapping
            for key in ("category", "task_type", "labels", "tags", "intent", "kind")
        )


def test_classifier_distinguishes_document_summary_from_coding():
    coding = _classify("Repair a Python CLI bug and add regression tests.")
    document = _classify(
        "Summarize a long leadership PDF into grounded Markdown sections with citations."
    )

    assert text_blob(coding) != text_blob(document)
    assert any(
        label in text_blob(document)
        for label in ("summary", "summar", "document", "pdf", "writing", "analysis")
    )
