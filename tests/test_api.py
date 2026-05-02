from __future__ import annotations

from fastapi.testclient import TestClient

from promptboost.main import app


def test_health_route(temp_db) -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_index_route(temp_db) -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "PromptBoost" in response.text


def test_boost_and_runs_routes(temp_db) -> None:
    client = TestClient(app)
    response = client.post(
        "/boost",
        json={
            "raw_prompt": "Fix the API bug and add tests without unrelated refactors.",
            "target_harness": "codex",
            "target_model": "gpt",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["detected_task_type"] == "coding_patch"
    assert body["result"]["input_adequacy"]["status"] in {"adequate", "needs_clarification"}
    assert "no_new_facts_result" in body["result"]
    assert body["result"]["no_new_facts_passed"] is True
    assert body["run"]["id"] == 1

    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 1


def test_promptgarage_samples_route(temp_db) -> None:
    client = TestClient(app)
    response = client.get("/api/promptgarage/samples?limit=2")

    assert response.status_code == 200
    assert "samples" in response.json()


def test_matrix_lab_page_and_deterministic_route(temp_db) -> None:
    client = TestClient(app)

    page = client.get("/matrix-lab")
    assert page.status_code == 200
    assert "Matrix Lab" in page.text

    response = client.post(
        "/api/matrix/deterministic",
        json={
            "raw_prompt": "Fix the API route, add tests, and avoid unrelated refactors.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["harnesses"]) == 10
    assert len(body["models"]) == 10
    assert len(body["rows"]) == 100
    assert body["rows"][0]["metrics"]["words"] > 0
