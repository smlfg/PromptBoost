from __future__ import annotations

from promptboost.pipeline import boost, classify_task, evaluate_no_new_facts, no_new_facts_check


def test_classification_covers_initial_task_types() -> None:
    assert classify_task("Fix this repository bug and add tests") == "coding_patch"
    assert classify_task("Research the evidence and cite sources") == "research_synthesis"
    assert classify_task("Summarize this PDF source faithfully") == "document_summary"
    assert classify_task("Extract these notes into JSON fields") == "structured_extraction"
    assert classify_task("Write a strategy memo with tradeoffs") == "strategic_memo"
    assert classify_task("Help me think through this") == "general_agent_task"
    assert classify_task("Implement the plan.") == "general_agent_task"


def test_adapter_selection_changes_prompt_structure() -> None:
    raw = "Fix the FastAPI route bug in this repo, add tests, and avoid unrelated refactors."
    codex = boost(raw, "codex", "gpt")
    hermes = boost(raw, "hermes", "minimax")

    assert codex.target_harness == "codex"
    assert codex.target_model == "gpt"
    assert "## 3. Harness Policy" in codex.boosted_prompt
    assert hermes.target_harness == "hermes"
    assert hermes.target_model == "minimax"
    assert "<promptboost>" in hermes.boosted_prompt
    assert "artifact ownership" in " ".join(hermes.adapter_rules["harness"]["rules"]).lower()


def test_underspecified_prompt_gets_missing_information_section() -> None:
    result = boost("Make it better", "codex", "gpt")

    assert "Missing information / assumptions" in result.boosted_prompt
    assert any("Missing information" in warning for warning in result.warnings)
    assert result.input_adequacy["status"] == "blocked"
    assert result.readiness_score is not None
    assert result.readiness_score.value <= 35


def test_default_deliverables_are_suggested_scaffolding_not_observed_intent() -> None:
    result = boost("Help me think through this", "codex", "gpt")

    assert result.task_spec["deliverables"] == []
    assert result.task_spec["suggested_scaffolding"]["deliverables"]
    assert "Suggested scaffolding, not user facts" in result.boosted_prompt


def test_structured_extraction_uses_json_only_output_contract() -> None:
    result = boost("Extract names and dates from the provided notes into JSON.", "codex", "gpt")

    assert result.detected_task_type == "structured_extraction"
    assert "Return valid JSON only" in result.boosted_prompt
    assert "no Markdown fence or prose wrapper" in result.boosted_prompt


def test_no_new_facts_guard_flags_domain_additions() -> None:
    warnings, passed = no_new_facts_check(
        "Summarize the provided notes.",
        "Summarize the provided notes about Kubernetes latency in Berlin.",
        {"harness": {"allowed_terms": []}},
    )

    assert not passed
    assert warnings


def test_no_new_facts_guard_allows_framework_structure() -> None:
    result = boost("Summarize the provided source text without adding facts.", "codex", "gpt")

    assert result.no_new_facts_passed


def test_german_pdf_prompt_extracts_deliverables_and_constraints() -> None:
    result = boost(
        "Fasse ein langes PDF quellentreu in Markdown zusammen und erstelle ein PDF-Render-Skript.",
        "codex",
        "gpt",
    )
    deliverables_blob = " ".join(result.task_spec["deliverables"]).lower()
    constraints_blob = " ".join(result.task_spec["constraints"]).lower()

    assert result.detected_task_type == "document_summary"
    assert "markdown" in deliverables_blob
    assert "pdf-render-skript" in deliverables_blob
    assert "quellentreu" in constraints_blob
    assert "No hard constraints were explicit" not in result.boosted_prompt
    assert result.no_new_facts_passed


def test_guard_splits_harmless_meta_from_suspected_facts() -> None:
    result = evaluate_no_new_facts(
        "Summarize the provided PDF.",
        "Fully summarize the provided PDF and add Kubernetes latency findings in Berlin.",
        {"harness": {"allowed_terms": []}},
    )

    assert not result.passed
    assert "fully" in result.harmless_meta_terms
    assert {"kubernetes", "latency", "berlin"}.issubset(set(result.suspected_new_facts))
