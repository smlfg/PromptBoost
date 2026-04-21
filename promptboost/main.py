"""FastAPI entrypoint for PromptBoost."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__
from .config import STATIC_DIR, TEMPLATES_DIR
from .db import init_db, list_boost_runs, save_boost_run
from .pipeline import boost
from .promptgarage import promptgarage_available, sample_prompts
from .rules.loader import RuleNotFoundError, available_harnesses, available_models
from .schemas import BoostRequest, BoostResult


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="PromptBoost", version=__version__, lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "harnesses": available_harnesses(),
            "models": available_models(),
        },
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "version": __version__,
        "promptgarage_available": promptgarage_available(),
        "harnesses": available_harnesses(),
        "models": available_models(),
    }


@app.post("/boost")
def boost_route(payload: BoostRequest) -> dict[str, object]:
    try:
        result = boost(payload.raw_prompt, payload.target_harness, payload.target_model)
    except RuleNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if isinstance(result, BoostResult):
        run = save_boost_run(result)
        return {"result": result.model_dump(), "run": run.model_dump()}
    if isinstance(result, dict):
        return {"result": result, "run": {"id": result.get("run_id")}}
    raise HTTPException(status_code=500, detail="Unexpected boost result")


@app.get("/api/runs")
def runs(limit: int = Query(default=12, ge=1, le=50)) -> dict[str, object]:
    return {"runs": [run.model_dump() for run in list_boost_runs(limit=limit)]}


@app.get("/api/promptgarage/samples")
def promptgarage_samples(
    limit: int = Query(default=6, ge=1, le=20),
    q: str | None = Query(default=None),
) -> dict[str, object]:
    return {"samples": [sample.model_dump() for sample in sample_prompts(limit=limit, query=q)]}
