from __future__ import annotations

import pytest

from conftest import import_any, install_fake_adapter, maybe_import, text_blob, unload_promptboost_modules


API_MODULES = (
    "promptboost.api",
    "promptboost.app",
    "promptboost.main",
    "promptboost.server",
)


def _fake_boost(*_args, **_kwargs):
    return {
        "run_id": "test-run",
        "boosted_prompt": "API boosted prompt without paid API calls.",
        "target_harness": _kwargs.get("target_harness") or _kwargs.get("harness") or "codex",
        "target_model": _kwargs.get("target_model") or _kwargs.get("model") or "test-model",
    }


def _patch_api_boost(monkeypatch):
    for module_name in (
        "promptboost",
        "promptboost.core",
        "promptboost.boost",
        "promptboost.api",
        "promptboost.app",
        "promptboost.main",
        "promptboost.server",
    ):
        module = maybe_import(module_name)
        if module is not None:
            monkeypatch.setattr(module, "boost", _fake_boost, raising=False)


def _load_app(monkeypatch):
    api_module = import_any(*API_MODULES)
    _patch_api_boost(monkeypatch)

    create_app = getattr(api_module, "create_app", None)
    if callable(create_app):
        return create_app()

    app = getattr(api_module, "app", None)
    if app is None:
        pytest.fail(
            "FastAPI module must expose either app or create_app() for route tests."
        )
    return app


def _request_first_existing(client, method: str, paths: list[str], **kwargs):
    for path in paths:
        response = getattr(client, method)(path, **kwargs)
        if response.status_code != 404:
            return path, response
    pytest.fail(f"None of the expected API routes exist: {', '.join(paths)}")


def test_api_health_route(promptboost_db_path, monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    unload_promptboost_modules()
    install_fake_adapter(monkeypatch, "API boosted prompt without paid API calls.")
    app = _load_app(monkeypatch)
    client = testclient.TestClient(app)

    path, response = _request_first_existing(
        client,
        "get",
        ["/health", "/healthz", "/api/health", "/"],
    )

    assert response.status_code == 200, path
    assert any(token in text_blob(response.json()) for token in ("ok", "healthy", "promptboost"))


def test_api_boost_route_returns_boosted_prompt(promptboost_db_path, monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    unload_promptboost_modules()
    install_fake_adapter(monkeypatch, "API boosted prompt without paid API calls.")
    app = _load_app(monkeypatch)
    client = testclient.TestClient(app)

    _path, response = _request_first_existing(
        client,
        "post",
        ["/boost", "/api/boost", "/v1/boost"],
        json={
            "raw_prompt": "Improve this API prompt without calling paid services.",
            "target_harness": "codex",
            "target_model": "test-model",
        },
    )

    assert response.status_code in (200, 201)
    assert "api boosted prompt without paid api calls" in text_blob(response.json())
