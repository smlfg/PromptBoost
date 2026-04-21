from __future__ import annotations

from fastapi.testclient import TestClient

from promptboost.db import list_boost_runs
from promptboost.main import app
from promptboost.rules.loader import available_models


EXPECTED_MODELS = {
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
}


def test_all_v2_model_adapters_are_available() -> None:
    assert EXPECTED_MODELS.issubset(set(available_models()))


def test_boost_route_accepts_all_v2_model_adapters(temp_db) -> None:
    client = TestClient(app)
    raw_prompt = (
        "Fasse ein langes PDF quellentreu in Markdown zusammen und erstelle "
        "ein PDF-Render-Skript."
    )

    for model in sorted(EXPECTED_MODELS):
        response = client.post(
            "/boost",
            json={
                "raw_prompt": raw_prompt,
                "target_harness": "codex",
                "target_model": model,
            },
        )

        assert response.status_code == 200, model
        result = response.json()["result"]
        assert result["target_model"] == model
        assert result["model_adapter"]["id"] == model
        assert result["score"]["overall"] > 0
        assert result["eval_plan"]["required_checks"]
        assert "Task Spec" in result["boosted_prompt"]


def test_boost_persists_full_v2_result_json(temp_db) -> None:
    client = TestClient(app)
    response = client.post(
        "/boost",
        json={
            "raw_prompt": "Extract these notes into a valid JSON object with missing fields as null.",
            "target_harness": "codex",
            "target_model": "qwen",
        },
    )

    assert response.status_code == 200
    runs = list_boost_runs(db_path=temp_db)
    assert len(runs) == 1
    assert runs[0].result["model_adapter"]["id"] == "qwen"
    assert runs[0].result["eval_plan"]["metadata_fields"]
    assert runs[0].score is not None


def test_health_reports_available_v2_adapters(temp_db) -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert EXPECTED_MODELS.issubset(set(body["models"]))
