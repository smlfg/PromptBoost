from __future__ import annotations

from pathlib import Path

from promptboost.adapters import HARNESS_ALIASES, select_adapter
from promptboost.config import RULES_DIR
from promptboost.pipeline import boost
from promptboost.rules.loader import available_harnesses, load_harness


NEW_HARNESSES = {
    "browser_research": {
        "aliases": ("browser", "browser_research"),
        "raw": "Research the latest evidence and cite sources with concrete dates.",
        "tokens": ("Browser Research", "Primary sources", "Currentness"),
    },
    "tool_calling_agent": {
        "aliases": ("tool_calling", "tool_calling_agent"),
        "raw": "Use the available tool output to answer, and do not invent observations.",
        "tokens": ("Tool Calling Agent", "Tool availability", "Tool outputs"),
    },
    "ci_debug_agent": {
        "aliases": ("ci", "ci_debug", "ci_debug_agent"),
        "raw": "Fix the failing CI check from the logs with a minimal patch.",
        "tokens": ("CI Debug Agent", "Failed CI check", "Flaky"),
    },
    "ide_assistant": {
        "aliases": ("ide", "ide_assistant"),
        "raw": "Patch the selected file in the IDE and preserve unrelated workspace changes.",
        "tokens": ("IDE Assistant", "Active file", "Workspace state"),
    },
    "cloud_sandbox": {
        "aliases": ("cloud", "cloud_sandbox"),
        "raw": "Run this task in a cloud sandbox and export reproducible artifacts.",
        "tokens": ("Cloud Sandbox", "Ephemeral filesystem", "Artifacts"),
    },
    "no_tools_chat": {
        "aliases": ("no_tools", "no_tools_chat"),
        "raw": "Answer from the provided context only and state assumptions.",
        "tokens": ("No-Tools Chat", "No browsing", "Assumptions"),
    },
    "multi_agent_orchestrator": {
        "aliases": ("multi_agent", "multi_agent_orchestrator"),
        "raw": "Coordinate multiple agents with separate ownership and merge their findings.",
        "tokens": ("Multi-Agent Orchestrator", "Dependency graph", "Merge authority"),
    },
}


REQUIRED_KEYS = {
    "id",
    "label",
    "summary",
    "section_style",
    "rules",
    "verification_gates",
    "allowed_terms",
}


def test_every_harness_alias_points_to_existing_rulepack() -> None:
    for alias, canonical in HARNESS_ALIASES.items():
        adapter = select_adapter(alias, "gpt")

        assert adapter["normalized_harness"] == canonical
        assert adapter["harness"]["id"] == canonical
        assert (RULES_DIR / "harnesses" / f"{canonical}.json").exists()


def test_all_harness_rulepacks_have_required_shape() -> None:
    for path in sorted((RULES_DIR / "harnesses").glob("*.json")):
        harness = load_harness(path.stem)

        assert REQUIRED_KEYS.issubset(harness), path.name
        assert harness["id"] == path.stem
        assert isinstance(harness["rules"], list) and harness["rules"]
        assert isinstance(harness["verification_gates"], list) and harness["verification_gates"]
        assert isinstance(harness["allowed_terms"], list) and harness["allowed_terms"]


def test_new_harnesses_are_available_to_the_ui_selector() -> None:
    available = set(available_harnesses())

    assert set(NEW_HARNESSES).issubset(available)
    assert "hermes" not in available


def test_new_harness_aliases_boost_end_to_end() -> None:
    for canonical, case in NEW_HARNESSES.items():
        for alias in case["aliases"]:
            result = boost(case["raw"], alias, "gpt")
            prompt_blob = result.boosted_prompt.lower()

            assert result.target_harness == canonical
            assert result.no_new_facts_passed, (canonical, alias, result.no_new_facts_result)
            for token in case["tokens"]:
                assert token.lower() in prompt_blob


def test_legacy_harnesses_still_work() -> None:
    cases = [
        ("codex", "gpt"),
        ("hermes", "minimax"),
        ("opencode", "minimax"),
        ("claude_code", "claude"),
    ]

    for harness, model in cases:
        result = boost("Fix a code bug and add tests without unrelated refactors.", harness, model)

        assert result.target_harness in {harness, "hermes", "opencode", "claude_code"}
        assert result.target_model in {model, "gpt", "minimax", "claude"}
        assert result.boosted_prompt
