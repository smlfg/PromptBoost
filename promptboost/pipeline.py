"""Deterministic PromptBoost pipeline."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from promptboost.adapters import normalize_harness, normalize_model
from promptboost.rules.loader import load_harness, load_model, load_task_types
from promptboost.schemas import (
    BoostResult,
    EvalPlan,
    EvidenceSummary,
    NoNewFactsResult,
    PromptScore,
    PromptScoreDimension,
    ReadinessScore,
    SourceRef,
    TaskType,
)


TASK_ORDER: tuple[TaskType, ...] = (
    "coding_patch",
    "research_synthesis",
    "document_summary",
    "structured_extraction",
    "strategic_memo",
    "general_agent_task",
)

GENERIC_ALLOWED_TERMS = {
    "acceptance",
    "adapter",
    "agentic",
    "added",
    "adequacy",
    "adequate",
    "allowed",
    "answer",
    "agent",
    "assumption",
    "assumptions",
    "available",
    "based",
    "before",
    "boosted",
    "beyond",
    "boundary",
    "boundaries",
    "breaking",
    "check",
    "checks",
    "clear",
    "candidate",
    "cap",
    "clarify",
    "claim",
    "claims",
    "clarification",
    "clarifications",
    "critical",
    "complete",
    "confirm",
    "confidence",
    "constraints",
    "context",
    "contract",
    "criteria",
    "current",
    "dates",
    "define",
    "details",
    "deliverable",
    "deliverables",
    "detected",
    "detected_type",
    "discovered",
    "domain",
    "evidence",
    "evidence_level",
    "existing",
    "explicit",
    "extra",
    "family",
    "facts",
    "fact_guard",
    "final",
    "fixed",
    "follow",
    "format",
    "functional",
    "gaps",
    "generated",
    "goal",
    "grounded",
    "guard",
    "guardrail",
    "guardrails",
    "harness",
    "harness_policy",
    "include",
    "improved",
    "information",
    "input",
    "inputs",
    "interdependency",
    "intent",
    "local",
    "language",
    "maintain",
    "missing",
    "missing_information_assumptions",
    "metrics",
    "model",
    "model_adapter",
    "must",
    "names",
    "new",
    "output",
    "open",
    "questions",
    "open_questions",
    "policy",
    "present",
    "preserve",
    "prompt",
    "promptboost",
    "provided",
    "quality",
    "raw",
    "readability",
    "raw_task_boundary",
    "readiness",
    "requirements",
    "reason",
    "reasons",
    "reduced",
    "refactored",
    "result",
    "return",
    "rules",
    "risk",
    "risk_flags",
    "runtime",
    "scope",
    "section",
    "separate",
    "separated",
    "separation",
    "short",
    "source",
    "sources",
    "specified",
    "state",
    "status",
    "success",
    "steps",
    "task",
    "target",
    "task_spec",
    "hypotheses",
    "hypothesis",
    "json",
    "observed",
    "scaffolding",
    "score",
    "there",
    "unchanged",
    "user",
    "validated",
    "suggested",
    "verification",
    "verification_gates",
    "warnings",
    "which",
    "without",
    "workflow",
}

HARMLESS_META_TERMS = {
    "before",
    "carefully",
    "complete",
    "completely",
    "completion",
    "comprehensive",
    "clarify",
    "clarification",
    "draft",
    "explicitly",
    "faithful",
    "fully",
    "grounding",
    "instructions",
    "reason",
    "reasons",
    "metasprache",
    "failure-risk",
    "level",
    "notes",
    "pass",
    "process",
    "requested",
    "self-check",
    "scaffolding",
    "state",
    "structure",
    "structured",
    "status",
    "summary",
    "score",
    "unsupported",
    "usable",
    "visible",
}

SUSPICIOUS_DOMAIN_TERMS = {
    "api",
    "authentication",
    "aws",
    "azure",
    "berlin",
    "database",
    "docker",
    "graphql",
    "kubernetes",
    "latency",
    "login",
    "oauth",
    "pinecone",
    "postgres",
    "queue",
    "queues",
    "redis",
    "s3",
    "search",
    "sqlite",
    "vector",
}

DELIVERABLE_SIGNALS = (
    "add ",
    "analyze",
    "build",
    "create",
    "deliver",
    "extract",
    "fix",
    "generate",
    "implement",
    "plan",
    "produce",
    "review",
    "return",
    "summarize",
    "write",
    "analysiere",
    "baue",
    "erstelle",
    "extrahiere",
    "erzeuge",
    "erkläre",
    "erklaere",
    "fasse",
    "gib",
    "liefere",
    "plane",
    "prüfe",
    "pruefe",
    "repariere",
    "schreibe",
)

CONSTRAINT_SIGNALS = (
    "avoid",
    "do not",
    "don't",
    "keep ",
    "must",
    "no ",
    "only",
    "source-faithful",
    "without",
    "belege",
    "belegt",
    "keine",
    "nicht",
    "nur",
    "ohne",
    "prüfe",
    "pruefe",
    "quellentreu",
    "unverändert",
    "unveraendert",
)


def boost(
    raw_prompt: str,
    target_harness: str = "codex",
    target_model: str = "gpt",
    agentic_task_draft: Any | None = None,
    agentic_metadata: Any | None = None,
) -> BoostResult:
    cleaned = raw_prompt.strip()
    if not cleaned:
        raise ValueError("raw_prompt is required")

    harness_id = normalize_harness(target_harness)
    model_id = normalize_model(target_model)
    harness = load_harness(harness_id)
    model = load_model(model_id)
    task_types = load_task_types()
    detected_task_type = classify_task(cleaned)
    task_rule = task_types[detected_task_type]
    task_spec = extract_task_spec(cleaned, detected_task_type, task_rule)
    input_adequacy = task_spec["input_adequacy"]

    adapter_rules = {
        "harness": harness,
        "model": model,
        "task_type": task_rule,
    }
    candidate_prompt = assemble_prompt(cleaned, task_spec, harness, model, task_rule)
    if agentic_task_draft is not None:
        candidate_prompt = append_agentic_task_draft(candidate_prompt, agentic_task_draft)
    guard_result = evaluate_no_new_facts(cleaned, candidate_prompt, adapter_rules)
    warnings = [*guard_result.warnings, *task_spec["warnings"]]
    evidence_summary = build_evidence_summary(harness, model)
    failure_modes = _dedupe_preserve_order(
        [*harness.get("failure_modes", []), *model.get("failure_modes", [])]
    )
    readiness_score = calculate_readiness_score(
        task_spec,
        harness,
        model,
        guard_result,
        input_adequacy,
    )
    prompt_score = PromptScore(
        overall=readiness_score.value,
        grade=_score_grade(readiness_score.value),
        source="prompt_shape_heuristic",
        dimensions=[
            PromptScoreDimension(
                key="structure",
                label="Prompt structure",
                value=min(5, max(1, round(readiness_score.value / 20))),
                evidence="Heuristic score from prompt shape, input adequacy, adapter rules, and guardrail sections.",
            )
        ],
        findings=readiness_score.findings,
    )

    return BoostResult(
        raw_prompt=cleaned,
        detected_task_type=detected_task_type,
        target_harness=harness["id"],
        target_model=model["id"],
        adapter_rules=adapter_rules,
        boosted_prompt=candidate_prompt,
        candidate_prompt=candidate_prompt,
        warnings=warnings,
        no_new_facts_passed=guard_result.passed,
        no_new_facts_result=guard_result,
        task_spec=task_spec,
        harness_policy=harness,
        model_adapter=model,
        eval_plan=build_eval_plan(task_rule, harness, model, failure_modes),
        score=prompt_score,
        evidence_summary=evidence_summary,
        failure_modes=failure_modes,
        readiness_score=readiness_score,
        matrix_cell={
            "harness": harness["id"],
            "model": model["id"],
            "label": f"{harness['label']} × {model['label']}",
        },
        input_adequacy=input_adequacy,
        agentic=agentic_metadata,
    )


def append_agentic_task_draft(candidate_prompt: str, task_draft: Any) -> str:
    draft = task_draft.model_dump() if hasattr(task_draft, "model_dump") else dict(task_draft)
    lines = [
        "",
        "## Agentic Task Draft",
        "Validated LLM assistive extraction. The raw task boundary remains authoritative.",
        f"- Task type: {draft.get('task_type', 'unknown')}",
        f"- Goal: {draft.get('goal', '')}",
        _bullet_block("Inputs", draft.get("inputs") or ["No inputs were identified by the LLM draft."]),
        _bullet_block("Deliverables", draft.get("deliverables") or ["No deliverables were identified by the LLM draft."]),
        _bullet_block("Constraints", draft.get("constraints") or ["No constraints were identified by the LLM draft."]),
        _bullet_block("Success criteria", draft.get("success_criteria") or ["No success criteria were identified by the LLM draft."]),
        _bullet_block("Open questions", draft.get("open_questions") or ["No open questions were identified by the LLM draft."]),
        _bullet_block("Assumptions", draft.get("assumptions") or ["No assumptions were identified by the LLM draft."]),
        _bullet_block("Risk flags", draft.get("risk_flags") or ["No risk flags were identified by the LLM draft."]),
    ]
    return "\n".join([candidate_prompt, *lines])


def build_evidence_summary(harness: dict[str, Any], model: dict[str, Any]) -> EvidenceSummary:
    levels = [_evidence_level(harness), _evidence_level(model)]
    level_order = {"hypothesis": 0, "inferred": 1, "documented": 2, "eval_backed": 3}
    combined_level = min(levels, key=lambda level: level_order.get(str(level), 0))
    sources = []
    for source in [*harness.get("sources", []), *model.get("sources", [])]:
        if isinstance(source, dict) and source.get("title") and source.get("url"):
            sources.append(SourceRef(**source))

    return EvidenceSummary(
        level=combined_level,
        claim_notes=[*harness.get("claim_notes", []), *model.get("claim_notes", [])],
        sources=sources,
        eval_run_ids=[*harness.get("eval_run_ids", []), *model.get("eval_run_ids", [])],
    )


def calculate_readiness_score(
    task_spec: dict[str, Any],
    harness: dict[str, Any],
    model: dict[str, Any],
    guard_result: NoNewFactsResult,
    input_adequacy: dict[str, Any],
) -> ReadinessScore:
    value = 45
    findings: list[str] = []

    if task_spec.get("deliverables"):
        value += 10
        findings.append("Observed deliverables are explicit in the raw prompt.")
    elif task_spec.get("suggested_scaffolding", {}).get("deliverables"):
        value += 3
        findings.append("Only suggested deliverable scaffolding is available; it is not treated as user intent.")
    if task_spec.get("quality_gates"):
        value += 10
        findings.append("Quality gates are present.")
    if harness.get("rules") and harness.get("verification_gates"):
        value += 15
        findings.append("Harness policy includes runtime rules and verification gates.")
    if model.get("rules"):
        value += 10
        findings.append("Model adapter includes prompt-shaping rules.")
    if _evidence_level(harness) and _evidence_level(model):
        value += 5
        findings.append("Evidence level is declared for both harness and model.")
    if guard_result.passed:
        value += 5
        findings.append("No-new-facts guard found no suspected domain additions.")
    else:
        value -= 10
        findings.append("No-new-facts guard requires review before use.")

    status = str(input_adequacy.get("status", "adequate"))
    score_cap = input_adequacy.get("score_cap")
    if status == "blocked":
        findings.append("Input adequacy is blocked; the candidate should be clarified before use.")
    elif status == "needs_clarification":
        findings.append("Input adequacy needs clarification; the candidate is usable only as a draft.")
    if isinstance(score_cap, int) and value > score_cap:
        value = score_cap
        findings.append(f"Readiness capped at {score_cap} by input adequacy.")

    return ReadinessScore(value=max(0, min(100, value)), findings=findings)


def build_eval_plan(
    task_rule: dict[str, Any],
    harness: dict[str, Any],
    model: dict[str, Any],
    failure_modes: list[str],
) -> EvalPlan:
    return EvalPlan(
        rubric=[
            {
                "key": "structure",
                "label": "Structure/guardrail heuristic",
                "description": "Checks whether the candidate prompt exposes task, runtime, model, and verification sections.",
            },
            {
                "key": "task_preservation",
                "label": "Task preservation",
                "description": "Checks that the raw task boundary remains visible.",
            },
            {
                "key": "evidence_visibility",
                "label": "Evidence visibility",
                "description": "Checks whether documented/inferred/hypothesis status is visible.",
            },
        ],
        expected_failure_modes=failure_modes,
        required_checks=[*task_rule.get("quality_gates", []), *harness.get("verification_gates", [])],
        metadata_fields=["harness", "model", "evidence_level", "sources", "readiness_score", "input_adequacy"],
    )


def _score_grade(value: int) -> str:
    if value >= 85:
        return "strong"
    if value >= 65:
        return "usable"
    if value >= 40:
        return "weak"
    return "blocked"


def _evidence_level(rule: dict[str, Any]) -> str:
    explicit = rule.get("evidence_level")
    if explicit:
        return str(explicit)
    evidence = str(rule.get("evidence", "")).lower()
    if evidence in {"strong", "strong_prior", "documented"}:
        return "documented"
    if evidence in {"medium", "moderate", "moderate_prior", "inferred", "partial"}:
        return "inferred"
    if evidence in {"eval", "eval_backed"}:
        return "eval_backed"
    return "hypothesis"


def classify_task(raw_prompt: str) -> TaskType:
    text = raw_prompt.lower()
    task_types = load_task_types()
    scores: Counter[str] = Counter()
    for task_type, config in task_types.items():
        for keyword in config.get("keywords", []):
            if re.search(rf"\b{re.escape(keyword.lower())}\b", text):
                scores[task_type] += 1

    if scores.get("coding_patch") and not _has_coding_context(raw_prompt):
        scores["coding_patch"] = 0
    if scores.get("strategic_memo") and not _has_strategic_context(raw_prompt):
        scores["strategic_memo"] = 0
    scores += Counter()

    if not scores:
        return "general_agent_task"

    best_score = max(scores.values())
    if best_score <= 0:
        return "general_agent_task"
    for task_type in TASK_ORDER:
        if scores[task_type] == best_score:
            return task_type
    return "general_agent_task"


def extract_task_spec(raw_prompt: str, task_type: TaskType, task_rule: dict[str, Any]) -> dict[str, Any]:
    goal = _first_sentence(raw_prompt)
    constraints = _extract_constraint_lines(raw_prompt)
    explicit_deliverables = _extract_deliverables(raw_prompt)
    suggested_deliverables = task_rule["default_deliverables"]

    missing: list[str] = []
    if len(_tokens(raw_prompt)) < 10:
        missing.append("The raw prompt is short and may not define full context.")
    if not _has_input_signal(raw_prompt):
        missing.append("Inputs, sources, or files are not fully specified.")
    if not constraints:
        missing.append("Hard constraints are not fully specified.")

    input_adequacy = assess_input_adequacy(
        raw_prompt=raw_prompt,
        task_type=task_type,
        explicit_deliverables=explicit_deliverables,
        constraints=constraints,
        missing_information=missing,
    )
    warnings = [f"Missing information: {item}" for item in missing]
    if input_adequacy["status"] != "adequate":
        warnings.append(f"Input adequacy: {input_adequacy['summary']}")
    return {
        "goal": goal,
        "task_type": task_type,
        "deliverables": explicit_deliverables,
        "observed_intent": {
            "goal": goal,
            "deliverables": explicit_deliverables,
            "constraints": constraints,
            "input_signals": _input_signals(raw_prompt),
        },
        "suggested_scaffolding": {
            "deliverables": suggested_deliverables,
            "quality_gates": task_rule["quality_gates"],
            "output_contract": _output_contract_lines(task_type),
        },
        "constraints": constraints,
        "quality_gates": task_rule["quality_gates"],
        "missing_information": missing,
        "input_adequacy": input_adequacy,
        "warnings": warnings,
    }


def assess_input_adequacy(
    raw_prompt: str,
    task_type: TaskType,
    explicit_deliverables: list[str],
    constraints: list[str],
    missing_information: list[str],
) -> dict[str, Any]:
    tokens = _tokens(raw_prompt)
    reasons: list[str] = []
    required_clarifications: list[str] = []

    if len(tokens) < 3:
        return _input_adequacy_result(
            status="blocked",
            score_cap=35,
            reasons=["The raw prompt is too short to preserve a reliable task boundary."],
            required_clarifications=[
                "Add the concrete target, available context, requested output, and hard constraints."
            ],
        )

    if _is_generic_placeholder(raw_prompt):
        return _input_adequacy_result(
            status="blocked",
            score_cap=35,
            reasons=["The raw prompt is a generic placeholder and does not define the task."],
            required_clarifications=[
                "State what should be changed or produced, where the input lives, and what success means."
            ],
        )

    if len(tokens) < 10:
        reasons.append("The raw prompt is short and may not define full context.")
        required_clarifications.append("Add concrete context, input location, and success criteria.")
    if not explicit_deliverables:
        reasons.append("No explicit deliverable was detected in the raw prompt.")
        required_clarifications.append("Name the exact output the harness should produce.")
    if _task_needs_source(task_type) and not _has_input_signal(raw_prompt):
        reasons.append("The task type usually needs a source, file, URL, note set, or other input boundary.")
        required_clarifications.append("Provide or point to the source material that should ground the answer.")
    if task_type == "coding_patch" and not _has_coding_context(raw_prompt):
        reasons.append("The prompt uses implementation language without a clear codebase or file context.")
        required_clarifications.append("Name the repository, files, failing behavior, or test target.")
    if not constraints:
        reasons.append("Hard constraints are not fully specified.")
        required_clarifications.append("List non-negotiable constraints such as scope, format, source use, or tests.")

    if reasons:
        return _input_adequacy_result(
            status="needs_clarification",
            score_cap=59,
            reasons=_dedupe_preserve_order([*reasons, *missing_information]),
            required_clarifications=_dedupe_preserve_order(required_clarifications),
        )

    return _input_adequacy_result(
        status="adequate",
        score_cap=None,
        reasons=[],
        required_clarifications=[],
    )


def assemble_prompt(
    raw_prompt: str,
    task_spec: dict[str, Any],
    harness: dict[str, Any],
    model: dict[str, Any],
    task_rule: dict[str, Any],
) -> str:
    if harness["id"] in {"opencode", "multi_agent_orchestrator"} or model["id"] == "minimax":
        return _assemble_guarded_prompt(raw_prompt, task_spec, harness, model, task_rule)
    if harness["id"] == "claude_code" or model["id"] == "claude":
        return _assemble_phased_prompt(raw_prompt, task_spec, harness, model, task_rule)
    return _assemble_concise_prompt(raw_prompt, task_spec, harness, model, task_rule)


def no_new_facts_check(
    raw_prompt: str,
    boosted_prompt: str,
    adapter_rules: dict[str, Any],
) -> tuple[list[str], bool]:
    result = evaluate_no_new_facts(raw_prompt, boosted_prompt, adapter_rules)
    return result.warnings, result.passed


def evaluate_no_new_facts(
    raw_prompt: str,
    boosted_prompt: str,
    adapter_rules: dict[str, Any] | None = None,
) -> NoNewFactsResult:
    raw_terms = set(_meaningful_tokens(raw_prompt))
    allowed_terms = set(GENERIC_ALLOWED_TERMS) | set(HARMLESS_META_TERMS)
    allowed_terms.update(raw_terms)
    for block in (adapter_rules or {}).values():
        allowed_terms.update(_rule_terms(block))

    boosted_terms = set(_meaningful_tokens(boosted_prompt))
    new_terms = boosted_terms - raw_terms
    unknown_terms = {
        term
        for term in new_terms - allowed_terms
        if not term.isdigit() and len(term) > 4
    }
    suspected = sorted(
        term for term in unknown_terms if _is_suspected_new_fact(term, boosted_prompt)
    )
    suspected.extend(_new_numeric_claims(raw_prompt, boosted_prompt))
    suspected = sorted(set(suspected))

    harmless = sorted(
        (new_terms & set(HARMLESS_META_TERMS)) | (unknown_terms - set(suspected))
    )
    if not suspected:
        return NoNewFactsResult(
            passed=True,
            harmless_meta_terms=harmless,
            suspected_new_facts=[],
            warnings=[],
        )

    sample = ", ".join(suspected[:12])
    return NoNewFactsResult(
        passed=False,
        harmless_meta_terms=harmless,
        suspected_new_facts=suspected,
        warnings=[f"No-new-facts guard: suspected new facts require review: {sample}"],
    )


def _assemble_concise_prompt(
    raw_prompt: str,
    task_spec: dict[str, Any],
    harness: dict[str, Any],
    model: dict[str, Any],
    task_rule: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Candidate Prompt",
            "",
            "## 1. Raw Task Boundary",
            raw_prompt,
            "",
            "## 2. Task Spec",
            f"- Detected type: {task_rule['label']}",
            f"- Goal: {task_spec['goal']}",
            _bullet_block("Observed deliverables from raw prompt", _observed_deliverable_lines(task_spec)),
            _bullet_block("Suggested scaffolding, not user facts", _suggested_deliverable_lines(task_spec)),
            _bullet_block("Constraints from raw prompt", task_spec["constraints"] or ["No hard constraints were explicit."]),
            _bullet_block("Missing information / assumptions", _missing_lines(task_spec)),
            _bullet_block("Input adequacy", _input_adequacy_lines(task_spec["input_adequacy"])),
            "",
            "## 3. Harness Policy",
            f"- Target harness: {harness['label']}",
            _bullet_block("Rules", harness["rules"]),
            "",
            "## 4. Model Adapter",
            f"- Target model family: {model['label']}",
            _bullet_block("Rules", model["rules"]),
            "",
            "## 5. Evidence and Failure-Risk Notes",
            f"- Harness evidence level: {_evidence_level(harness)}",
            f"- Model evidence level: {_evidence_level(model)}",
            _bullet_block("Failure-risk hypotheses", [*harness.get("failure_modes", []), *model.get("failure_modes", [])] or ["No profile-specific failure modes declared."]),
            "",
            "## 6. Verification Before Final Output",
            _bullet_block("Quality gates", task_spec["quality_gates"] + harness["verification_gates"]),
            "- Confirm that the final answer adds no new task facts beyond the raw prompt.",
            "",
            "## 7. Final Output Contract",
            _bullet_block("Output contract", _output_contract_lines(task_spec["task_type"])),
        ]
    )


def _assemble_guarded_prompt(
    raw_prompt: str,
    task_spec: dict[str, Any],
    harness: dict[str, Any],
    model: dict[str, Any],
    task_rule: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "<promptboost>",
            "  <raw_task_boundary>",
            _indent(raw_prompt, 4),
            "  </raw_task_boundary>",
            "",
            "  <!-- Task Spec -->",
            "  <task_spec>",
            f"    <detected_type>{task_rule['label']}</detected_type>",
            f"    <goal>{task_spec['goal']}</goal>",
            _xml_list("observed_deliverables", _observed_deliverable_lines(task_spec), 4),
            _xml_list("suggested_scaffolding_not_user_facts", _suggested_deliverable_lines(task_spec), 4),
            _xml_list("constraints", task_spec["constraints"] or ["No hard constraints were explicit."], 4),
            _xml_list("missing_information_assumptions", _missing_lines(task_spec), 4),
            _xml_list("input_adequacy", _input_adequacy_lines(task_spec["input_adequacy"]), 4),
            "  </task_spec>",
            "",
            "  <harness_policy>",
            f"    <target>{harness['label']}</target>",
            _xml_list("rules", harness["rules"], 4),
            _xml_list("verification_gates", harness["verification_gates"], 4),
            "  </harness_policy>",
            "",
            "  <model_adapter>",
            f"    <target>{model['label']}</target>",
            _xml_list("rules", model["rules"], 4),
            "  </model_adapter>",
            "",
            "  <evidence>",
            f"    <harness_level>{_evidence_level(harness)}</harness_level>",
            f"    <model_level>{_evidence_level(model)}</model_level>",
            _xml_list("failure_risk_hypotheses", [*harness.get("failure_modes", []), *model.get("failure_modes", [])] or ["No profile-specific failure modes declared."], 4),
            "  </evidence>",
            "",
            "  <fact_guard>",
            "    <rule>Do not add domain facts, source claims, files, requirements, dates, names, metrics, or constraints not present in the raw task.</rule>",
            "    <rule>List missing information instead of filling gaps with plausible details.</rule>",
            "    <rule>Separate supported claims from assumptions before final output.</rule>",
            "  </fact_guard>",
            "",
            "  <output_contract>",
            _xml_list("rules", _output_contract_lines(task_spec["task_type"]), 4),
            "  </output_contract>",
            "</promptboost>",
        ]
    )


def _assemble_phased_prompt(
    raw_prompt: str,
    task_spec: dict[str, Any],
    harness: dict[str, Any],
    model: dict[str, Any],
    task_rule: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Candidate Prompt",
            "",
            "<context>",
            raw_prompt,
            "</context>",
            "",
            "<method>",
            "Task Spec:",
            f"Detected task type: {task_rule['label']}",
            f"Goal: {task_spec['goal']}",
            _bullet_block("Observed deliverables from raw prompt", _observed_deliverable_lines(task_spec)),
            _bullet_block("Suggested scaffolding, not user facts", _suggested_deliverable_lines(task_spec)),
            _bullet_block("Semantic phases", ["Ground in the provided context", "Work through the requested deliverables", "Review uncertainty and unsupported claims", "Return the final output"]),
            "</method>",
            "",
            "<rules>",
            _bullet_block("Harness rules", harness["rules"]),
            _bullet_block("Model rules", model["rules"]),
            _bullet_block("Missing information / assumptions", _missing_lines(task_spec)),
            _bullet_block("Input adequacy", _input_adequacy_lines(task_spec["input_adequacy"])),
            _bullet_block("Evidence levels", [f"Harness: {_evidence_level(harness)}", f"Model: {_evidence_level(model)}"]),
            _bullet_block("Failure-risk hypotheses", [*harness.get("failure_modes", []), *model.get("failure_modes", [])] or ["No profile-specific failure modes declared."]),
            "</rules>",
            "",
            "<output>",
            _bullet_block("Observed deliverables from raw prompt", _observed_deliverable_lines(task_spec)),
            _bullet_block("Suggested scaffolding, not user facts", _suggested_deliverable_lines(task_spec)),
            _bullet_block("Quality gates", task_spec["quality_gates"] + harness["verification_gates"]),
            _bullet_block("Output contract", _output_contract_lines(task_spec["task_type"])),
            "</output>",
        ]
    )


def _observed_deliverable_lines(task_spec: dict[str, Any]) -> list[str]:
    return task_spec["deliverables"] or [
        "No explicit deliverable was detected in the raw prompt; use the suggested scaffolding only as a draft structure."
    ]


def _suggested_deliverable_lines(task_spec: dict[str, Any]) -> list[str]:
    return task_spec.get("suggested_scaffolding", {}).get("deliverables", [])


def _input_adequacy_lines(input_adequacy: dict[str, Any]) -> list[str]:
    lines = [
        f"Status: {input_adequacy.get('status', 'unknown')}",
        str(input_adequacy.get("summary", "No input adequacy summary.")),
    ]
    score_cap = input_adequacy.get("score_cap")
    if score_cap is not None:
        lines.append(f"Readiness score cap: {score_cap}")
    reasons = input_adequacy.get("reasons") or []
    if reasons:
        lines.extend(f"Reason: {reason}" for reason in reasons)
    clarifications = input_adequacy.get("required_clarifications") or []
    if clarifications:
        lines.extend(f"Clarify: {item}" for item in clarifications)
    return lines


def _output_contract_lines(task_type: TaskType) -> list[str]:
    if task_type == "structured_extraction":
        return [
            "Return valid JSON only, with no Markdown fence or prose wrapper.",
            "Use null or an explicit empty list for missing values; do not invent fields.",
            "If the requested schema is missing, include schema gaps in a _missing_information field.",
        ]
    return [
        "Return the requested result in the format implied by the raw prompt.",
        "Include a short verification summary unless the raw prompt forbids extra prose.",
    ]


def _input_adequacy_result(
    status: str,
    score_cap: int | None,
    reasons: list[str],
    required_clarifications: list[str],
) -> dict[str, Any]:
    if status == "adequate":
        summary = "The raw prompt has enough observable task shape for a harness draft."
    elif status == "blocked":
        summary = "The raw prompt is too underspecified to prefer the boosted prompt without clarification."
    else:
        summary = "The boosted prompt is a structured draft, but clarification would materially improve reliability."

    return {
        "status": status,
        "score_cap": score_cap,
        "usable_without_clarification": status == "adequate",
        "summary": summary,
        "reasons": reasons,
        "required_clarifications": required_clarifications,
    }


def _first_sentence(raw_prompt: str) -> str:
    compact = " ".join(raw_prompt.split())
    parts = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)
    return parts[0][:300]


def _extract_constraint_lines(raw_prompt: str) -> list[str]:
    constraints: list[str] = []
    for line in _prompt_clauses(raw_prompt):
        stripped = line.strip(" -\t")
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in CONSTRAINT_SIGNALS):
            constraints.append(stripped)
    return _dedupe_preserve_order(constraints)[:8]


def _extract_deliverables(raw_prompt: str) -> list[str]:
    deliverables: list[str] = []
    for line in _prompt_clauses(raw_prompt):
        stripped = line.strip(" -\t")
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in DELIVERABLE_SIGNALS):
            deliverables.append(stripped)
    return _dedupe_preserve_order(deliverables)[:8]


def _has_input_signal(raw_prompt: str) -> bool:
    lowered = raw_prompt.lower()
    return bool(_input_signals(raw_prompt))


def _input_signals(raw_prompt: str) -> list[str]:
    lowered = raw_prompt.lower()
    signals = [
        marker
        for marker in (
            "file",
            "db",
            "database",
            "repo",
            "repository",
            "source",
            "text",
            "document",
            "notes",
            "pdf",
            "url",
            "path",
            "/",
            ".py",
            ".js",
            ".ts",
            ".md",
            ".json",
            "attached",
            "provided",
            "given",
            "vorhanden",
            "bereitgestellt",
            "quelle",
            "datei",
            "notizen",
        )
        if marker in lowered
    ]
    if re.search(r"```[a-z0-9_-]*\n", raw_prompt, flags=re.IGNORECASE):
        signals.append("code_fence")
    return _dedupe_preserve_order(signals)


def _has_coding_context(raw_prompt: str) -> bool:
    lowered = raw_prompt.lower()
    if re.search(r"\.(?:py|js|ts|tsx|jsx|json|yaml|yml|sql|html|css)\b", lowered):
        return True
    markers = (
        "add tests",
        "backend",
        "bug",
        "build error",
        "class ",
        "code",
        "codebase",
        "commit",
        "component",
        "debug",
        "endpoint",
        "error",
        "failing",
        "fix",
        "frontend",
        "function",
        "git",
        "issue",
        "module",
        "patch",
        "pytest",
        "repo",
        "repository",
        "route",
        "stack trace",
        "test failure",
        "testlauf",
        "test",
        "tests",
        "datei",
        "fehler",
        "funktion",
        "klasse",
        "modul",
        "repariere",
    )
    return any(marker in lowered for marker in markers)


def _has_strategic_context(raw_prompt: str) -> bool:
    lowered = raw_prompt.lower()
    markers = (
        "abwägung",
        "abwaegung",
        "strategy",
        "strategic",
        "strategie",
        "memo",
        "roadmap",
        "decision",
        "entscheidung",
        "tradeoff",
        "trade-off",
        "market",
        "positioning",
        "prioritize",
        "prioritise",
        "priorisiere",
    )
    return any(marker in lowered for marker in markers)


def _task_needs_source(task_type: TaskType) -> bool:
    return task_type in {"document_summary", "research_synthesis", "structured_extraction"}


def _is_generic_placeholder(raw_prompt: str) -> bool:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", raw_prompt.lower(), flags=re.UNICODE)).strip()
    placeholders = {
        "do it",
        "do this",
        "erledige das",
        "fix it",
        "help",
        "help me",
        "implement it",
        "improve it",
        "mach das besser",
        "mach es besser",
        "make it better",
        "make this better",
        "please fix",
        "please help",
        "verbessere es",
    }
    return normalized in placeholders


def _missing_lines(task_spec: dict[str, Any]) -> list[str]:
    missing = task_spec["missing_information"]
    if missing:
        return missing
    return ["No extra assumptions are allowed; use only the raw prompt and discovered context."]


def _bullet_block(title: str, items: list[str]) -> str:
    lines = [f"{title}:"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _xml_list(name: str, items: list[str], spaces: int) -> str:
    pad = " " * spaces
    lines = [f"{pad}<{name}>"]
    lines.extend(f"{pad}  <item>{item}</item>" for item in items)
    lines.append(f"{pad}</{name}>")
    return "\n".join(lines)


def _indent(value: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in value.splitlines())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_][\w./-]*", value, flags=re.UNICODE)


def _meaningful_tokens(value: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[^\W\d_][\w.-]*", value.lower(), flags=re.UNICODE):
        for normalized in _token_variants(token):
            if len(normalized) > 3 and normalized not in {"that", "then", "with", "from", "into", "only", "were", "what", "when", "where"}:
                tokens.append(normalized)
    return tokens


def _rule_terms(value: Any) -> set[str]:
    if isinstance(value, dict):
        terms: set[str] = set()
        for child in value.values():
            terms.update(_rule_terms(child))
        return terms
    if isinstance(value, list):
        terms = set()
        for child in value:
            terms.update(_rule_terms(child))
        return terms
    if isinstance(value, str):
        return set(_meaningful_tokens(value))
    return set()


def _prompt_clauses(raw_prompt: str) -> list[str]:
    clauses: list[str] = []
    for line in re.split(r"[\n.;:]+", raw_prompt):
        stripped = line.strip(" -\t")
        if not stripped:
            continue
        parts = re.split(
            r"\s+(?:und|and)\s+(?=(?:analysiere|erstelle|baue|schreibe|liefere|gib|erzeuge|extrahiere|fasse|plane|prüfe|pruefe|repariere|analyze|create|build|write|deliver|extract|fix|generate|plan|produce|return|review|summarize|implement|add)\b)",
            stripped,
            flags=re.IGNORECASE,
        )
        clauses.extend(part.strip(" -\t") for part in parts if part.strip(" -\t"))
    return clauses


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _token_variants(token: str) -> list[str]:
    stripped = token.strip("._-/")
    if not stripped:
        return []
    variants = [stripped]
    variants.extend(part for part in re.split(r"[._/-]+", stripped) if part)
    return variants


def _is_suspected_new_fact(term: str, boosted_prompt: str) -> bool:
    if term in SUSPICIOUS_DOMAIN_TERMS:
        return True
    if any(char.isdigit() for char in term):
        return True

    forms = _case_forms(boosted_prompt).get(term, [])
    for form in forms:
        if form.isupper() and len(form) > 2:
            return True
        if form[:1].isupper() and form[1:].islower() and term not in GENERIC_ALLOWED_TERMS:
            return True
    return False


def _new_numeric_claims(raw_prompt: str, boosted_prompt: str) -> list[str]:
    raw_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", raw_prompt))
    boosted_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", boosted_prompt))
    suspicious = []
    for number in boosted_numbers - raw_numbers:
        if number.endswith("%") or len(number) >= 3:
            suspicious.append(number)
    return suspicious


def _case_forms(value: str) -> dict[str, list[str]]:
    forms: dict[str, list[str]] = {}
    for token in re.findall(r"[^\W\d_][\w.-]*", value, flags=re.UNICODE):
        for variant in _token_variants(token):
            forms.setdefault(variant.lower(), []).append(variant)
    return forms
