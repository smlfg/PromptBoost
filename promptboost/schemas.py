"""Shared data contracts for the PromptBoost pipeline and API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskType = Literal[
    "coding_patch",
    "research_synthesis",
    "document_summary",
    "structured_extraction",
    "strategic_memo",
    "general_agent_task",
]

HarnessId = Literal[
    "codex",
    "opencode",
    "claude_code",
    "browser_research",
    "tool_calling_agent",
    "ci_debug_agent",
    "ide_assistant",
    "cloud_sandbox",
    "no_tools_chat",
    "multi_agent_orchestrator",
]
ModelId = Literal[
    "gpt",
    "claude",
    "gemini",
    "kimi",
    "glm",
    "qwen",
    "deepseek",
    "mistral",
    "minimax",
    "llama",
]
EvidenceLevel = Literal["documented", "inferred", "hypothesis", "eval_backed"]


class BoostRequest(BaseModel):
    raw_prompt: str = Field(min_length=1)
    target_harness: str = "codex"
    target_model: str = "gpt"


class NoNewFactsResult(BaseModel):
    passed: bool
    harmless_meta_terms: list[str] = Field(default_factory=list)
    suspected_new_facts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PromptScoreDimension(BaseModel):
    key: str
    label: str
    value: int
    evidence: str


class PromptScore(BaseModel):
    overall: int
    grade: Literal["blocked", "weak", "usable", "strong"]
    source: str = "prompt_shape_heuristic"
    dimensions: list[PromptScoreDimension] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)


class EvalPlan(BaseModel):
    rubric: list[dict[str, Any]] = Field(default_factory=list)
    expected_failure_modes: list[str] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    metadata_fields: list[str] = Field(default_factory=list)


class SourceRef(BaseModel):
    title: str
    url: str
    publisher: str | None = None
    accessed_at: str | None = None


class EvidenceSummary(BaseModel):
    level: EvidenceLevel
    claim_notes: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    eval_run_ids: list[str] = Field(default_factory=list)


class ReadinessScore(BaseModel):
    label: str = "Local Readiness Score"
    value: int
    max_value: int = 100
    disclaimer: str = (
        "This heuristic checks prompt structure and guardrails only; it does not measure "
        "model quality, output faithfulness, runtime, or cost."
    )
    findings: list[str] = Field(default_factory=list)


class BoostResult(BaseModel):
    raw_prompt: str
    detected_task_type: TaskType
    target_harness: str
    target_model: str
    adapter_rules: dict[str, Any]
    boosted_prompt: str
    candidate_prompt: str | None = None
    warnings: list[str]
    no_new_facts_passed: bool
    no_new_facts_result: NoNewFactsResult | None = None
    task_spec: dict[str, Any] = Field(default_factory=dict)
    harness_policy: dict[str, Any] = Field(default_factory=dict)
    model_adapter: dict[str, Any] = Field(default_factory=dict)
    eval_plan: EvalPlan | None = None
    score: PromptScore | None = None
    evidence_summary: EvidenceSummary | None = None
    failure_modes: list[str] = Field(default_factory=list)
    readiness_score: ReadinessScore | None = None
    matrix_cell: dict[str, str] = Field(default_factory=dict)
    input_adequacy: dict[str, Any] = Field(default_factory=dict)


class BoostRun(BaseModel):
    id: int
    created_at: str
    raw_prompt: str
    detected_task_type: str
    target_harness: str
    target_model: str
    adapter_rules: dict[str, Any]
    boosted_prompt: str
    no_new_facts_passed: bool
    warnings: list[str]
    result: dict[str, Any] = Field(default_factory=dict)
    score: PromptScore | None = None


class PromptGarageSample(BaseModel):
    id: int
    title: str
    content: str
    source: str | None = None
    tags: str | None = None
    model: str | None = None
    rating: int | None = None
    is_starred: bool = False
    updated_at: str | None = None
