"""Matrix evaluation helpers for PromptBoost."""

from __future__ import annotations

import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from promptboost.pipeline import boost, evaluate_no_new_facts
from promptboost.rules.loader import available_harnesses, available_models


DEFAULT_MATRIX_PROMPT = (
    "Fix the FastAPI /boost/agentic route in personalPromptBoost so it persists "
    "agentic metadata in result_json, add pytest route tests, preserve /boost "
    "deterministic behavior, and avoid unrelated refactors."
)

HERMES_ENV_PATH = Path("/home/smlflg/.hermes/.env")


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def section_count(text: str) -> int:
    return len(re.findall(r"(?m)^#{1,3}\s+|^\s*<[a-zA-Z_][^>]*>\s*$", text))


def bullet_count(text: str) -> int:
    return len(re.findall(r"(?m)^\s*[-*]\s+", text))


def structure_score(text: str) -> int:
    lower = text.lower()
    checks = [
        "raw task" in lower or "raw_task_boundary" in lower,
        "task spec" in lower or "task_spec" in lower,
        "harness" in lower,
        "model" in lower,
        "verification" in lower or "quality gate" in lower,
        "output contract" in lower or "output_contract" in lower,
        "missing" in lower or "assumption" in lower,
        "no new" in lower or "unsupported" in lower,
    ]
    return sum(1 for check in checks if check)


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return round(SequenceMatcher(None, left, right).ratio(), 4)


def metrics(text: str) -> dict[str, int]:
    return {
        "chars": len(text),
        "words": word_count(text),
        "sections": section_count(text),
        "bullets": bullet_count(text),
    }


def deterministic_matrix(raw_prompt: str) -> dict[str, Any]:
    harnesses = available_harnesses()
    models = available_models()
    rows = []
    for harness in harnesses:
        for model in models:
            result = boost(raw_prompt, harness, model)
            prompt = result.boosted_prompt
            rows.append(
                {
                    "harness": harness,
                    "model": model,
                    "score": result.score.overall if result.score else None,
                    "grade": result.score.grade if result.score else None,
                    "input_adequacy": result.input_adequacy.get("status"),
                    "no_new_facts_passed": result.no_new_facts_passed,
                    "structure_score": structure_score(prompt),
                    "metrics": metrics(prompt),
                    "preview": prompt[:900],
                }
            )
    return {
        "prompt": raw_prompt,
        "harnesses": harnesses,
        "models": models,
        "rows": rows,
    }


def full_agentic_cell(raw_prompt: str, harness: str, model: str) -> dict[str, Any]:
    deterministic = boost(raw_prompt, harness, model)
    deterministic_prompt = deterministic.boosted_prompt
    started = time.perf_counter()
    agentic_prompt = call_minimax_full_composer(raw_prompt, harness, model, deterministic_prompt)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    guard = evaluate_no_new_facts(raw_prompt, agentic_prompt, deterministic.adapter_rules)
    det_metrics = metrics(deterministic_prompt)
    agentic_metrics = metrics(agentic_prompt)

    return {
        "prompt": raw_prompt,
        "harness": harness,
        "model": model,
        "deterministic": {
            "score": deterministic.score.overall if deterministic.score else None,
            "grade": deterministic.score.grade if deterministic.score else None,
            "no_new_facts_passed": deterministic.no_new_facts_passed,
            "structure_score": structure_score(deterministic_prompt),
            "metrics": det_metrics,
            "preview": deterministic_prompt[:1200],
        },
        "agentic_full": {
            "no_new_facts_passed": guard.passed,
            "suspected_new_facts": guard.suspected_new_facts,
            "warnings": guard.warnings,
            "structure_score": structure_score(agentic_prompt),
            "metrics": agentic_metrics,
            "elapsed_ms": elapsed_ms,
            "preview": agentic_prompt[:1600],
        },
        "diff": {
            "char_delta": agentic_metrics["chars"] - det_metrics["chars"],
            "word_delta": agentic_metrics["words"] - det_metrics["words"],
            "section_delta": agentic_metrics["sections"] - det_metrics["sections"],
            "bullet_delta": agentic_metrics["bullets"] - det_metrics["bullets"],
            "structure_delta": structure_score(agentic_prompt) - structure_score(deterministic_prompt),
            "similarity": similarity(deterministic_prompt, agentic_prompt),
        },
    }


def call_minimax_full_composer(raw_prompt: str, harness: str, model: str, scaffold: str) -> str:
    api_key = minimax_api_key()
    if not api_key:
        raise RuntimeError("MiniMax API key is not configured.")

    system = (
        "You are PromptBoost's full agentic prompt composer. You receive a raw user "
        "prompt and a deterministic harness/model scaffold. Return one final "
        "harness-ready boosted prompt only. Preserve the raw task boundary as the "
        "source of truth. Do not add files, facts, dates, requirements, technologies, "
        "metrics, constraints, or acceptance criteria that are not in the raw prompt "
        "or scaffold. If information is missing, list it as missing information "
        "instead of guessing. Keep the target harness and model adapter visible."
    )
    user = (
        f"Target harness: {harness}\n"
        f"Target model profile: {model}\n\n"
        "<raw_user_prompt>\n"
        f"{raw_prompt}\n"
        "</raw_user_prompt>\n\n"
        "<deterministic_combo_scaffold>\n"
        f"{scaffold}\n"
        "</deterministic_combo_scaffold>\n\n"
        "Write the final boosted prompt for this exact harness/model combination."
    )
    payload = {
        "model": os.environ.get("PROMPTBOOST_MATRIX_LLM_MODEL", "MiniMax-M2.7"),
        "temperature": 0,
        "max_tokens": 3000,
        "reasoning_split": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    response = httpx.post(
        os.environ.get("PROMPTBOOST_MATRIX_LLM_BASE_URL", "https://api.minimax.io/v1").rstrip("/")
        + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=float(os.environ.get("PROMPTBOOST_MATRIX_LLM_TIMEOUT_SECONDS", "60")),
    )
    response.raise_for_status()
    data = response.json()
    return strip_think(data["choices"][0]["message"]["content"])


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def minimax_api_key() -> str | None:
    explicit = os.environ.get("PROMPTBOOST_MATRIX_LLM_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if explicit:
        return explicit
    if not HERMES_ENV_PATH.exists():
        return None
    for line in HERMES_ENV_PATH.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "MINIMAX_API_KEY":
            return value.strip().strip("'\"") or None
    return None
