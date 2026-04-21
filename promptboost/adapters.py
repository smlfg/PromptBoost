"""Adapter selection helpers."""

from __future__ import annotations

from typing import Any

from promptboost.rules.loader import load_harness, load_model


HARNESS_ALIASES = {
    "codex": "codex",
    "hermes": "hermes",
    "claude": "claude_code",
    "claude-code": "claude_code",
    "claude_code": "claude_code",
    "opencode": "opencode",
    "open_code": "opencode",
    "browser": "browser_research",
    "browser_research": "browser_research",
    "tool_calling": "tool_calling_agent",
    "tool_calling_agent": "tool_calling_agent",
    "ci": "ci_debug_agent",
    "ci_debug": "ci_debug_agent",
    "ci_debug_agent": "ci_debug_agent",
    "ide": "ide_assistant",
    "ide_assistant": "ide_assistant",
    "cloud": "cloud_sandbox",
    "cloud_sandbox": "cloud_sandbox",
    "no_tools": "no_tools_chat",
    "no_tools_chat": "no_tools_chat",
    "multi_agent": "multi_agent_orchestrator",
    "multi_agent_orchestrator": "multi_agent_orchestrator",
}

MODEL_ALIASES = {
    "gpt": "gpt",
    "gpt-5.4": "gpt",
    "gpt54": "gpt",
    "openai": "gpt",
    "minimax": "minimax",
    "minimax-m2.7": "minimax",
    "m2.7": "minimax",
    "m2": "minimax",
    "claude": "claude",
    "opus": "claude",
    "opus-4.6": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "kimi": "kimi",
    "moonshot": "kimi",
    "qwen": "qwen",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "glm": "glm",
    "zhipu": "glm",
    "llama": "llama",
    "meta-llama": "llama",
}


def normalize_harness(target_harness: str) -> str:
    key = target_harness.strip().lower().replace(" ", "_")
    if key not in HARNESS_ALIASES:
        raise ValueError(f"Unknown harness: {target_harness}")
    return HARNESS_ALIASES[key]


def normalize_model(target_model: str) -> str:
    key = target_model.strip().lower().replace(" ", "-")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model: {target_model}")
    return MODEL_ALIASES[key]


def select_adapter(target_harness: str = "codex", target_model: str = "gpt", **_: Any) -> dict[str, Any]:
    harness_id = normalize_harness(target_harness)
    model_id = normalize_model(target_model)
    return {
        "requested_harness": target_harness,
        "requested_model": target_model,
        "target_harness": target_harness.lower(),
        "target_model": target_model.lower(),
        "normalized_harness": harness_id,
        "normalized_model": model_id,
        "harness": load_harness(harness_id),
        "model": load_model(model_id),
    }


get_adapter = select_adapter
resolve_adapter = select_adapter
