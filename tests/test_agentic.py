from __future__ import annotations

from fastapi.testclient import TestClient

from promptboost.agentic import AgenticPrepassError, boost_agentic
from promptboost.main import app
from promptboost.schemas import AgenticTaskDraft


class FakeLLM:
    def __init__(
        self,
        draft: AgenticTaskDraft | None = None,
        raw_output: str | None = None,
        error: Exception | None = None,
        model: str = "fake-agentic-model",
    ) -> None:
        self.draft = draft
        self.raw_output = raw_output
        self.error = error
        self.model = model

    def create_task_draft(self, raw_prompt: str, target_harness: str, target_model: str):
        if self.error:
            raise self.error
        assert self.draft is not None
        return self.draft, self.raw_output or self.draft.model_dump_json(), self.model


def _safe_draft() -> AgenticTaskDraft:
    return AgenticTaskDraft(
        task_type="coding_patch",
        goal="Fix the API bug and add tests without unrelated refactors.",
        inputs=["API bug", "tests"],
        deliverables=["Fix the API bug", "add tests"],
        constraints=["without unrelated refactors"],
        success_criteria=["tests"],
        open_questions=[],
        assumptions=[],
        risk_flags=[],
        confidence=0.86,
    )


def test_agentic_prepass_adds_validated_draft(temp_db) -> None:
    result = boost_agentic(
        "Fix the API bug and add tests without unrelated refactors.",
        "codex",
        "gpt",
        llm_client=FakeLLM(_safe_draft()),
    )

    assert result.agentic is not None
    assert result.agentic.status == "accepted"
    assert result.no_new_facts_passed is True
    assert "## Agentic Task Draft" in result.boosted_prompt
    assert result.agentic.task_draft is not None
    assert result.agentic.task_draft.confidence == 0.86


def test_agentic_prepass_falls_back_on_llm_error(temp_db) -> None:
    result = boost_agentic(
        "Summarize the provided source text without adding facts.",
        "codex",
        "gpt",
        llm_client=FakeLLM(error=AgenticPrepassError("timeout")),
    )

    assert result.agentic is not None
    assert result.agentic.status == "fallback"
    assert "## Agentic Task Draft" not in result.boosted_prompt
    assert any("Agentic PromptBoost fallback" in warning for warning in result.warnings)


def test_agentic_prepass_discards_new_fact_draft(temp_db) -> None:
    draft = AgenticTaskDraft(
        task_type="document_summary",
        goal="Summarize the provided notes about Kubernetes latency in Berlin.",
        inputs=["provided notes"],
        deliverables=["Summarize the provided notes"],
        constraints=[],
        success_criteria=[],
        open_questions=[],
        assumptions=[],
        risk_flags=["Kubernetes latency in Berlin"],
        confidence=0.5,
    )

    result = boost_agentic(
        "Summarize the provided notes.",
        "codex",
        "gpt",
        llm_client=FakeLLM(draft),
    )

    assert result.agentic is not None
    assert result.agentic.status == "fallback"
    assert result.agentic.reason == "agentic_no_new_facts_failed"
    assert "## Agentic Task Draft" not in result.boosted_prompt


def test_agentic_route_falls_back_when_api_key_missing(temp_db, monkeypatch) -> None:
    monkeypatch.delenv("PROMPTBOOST_LLM_API_KEY", raising=False)
    client = TestClient(app)

    response = client.post(
        "/boost/agentic",
        json={
            "raw_prompt": "Fix the API bug and add tests without unrelated refactors.",
            "target_harness": "codex",
            "target_model": "gpt",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["agentic"]["status"] == "fallback"
    assert "PROMPTBOOST_LLM_API_KEY" in body["result"]["agentic"]["reason"]
    assert body["run"]["id"] == 1
