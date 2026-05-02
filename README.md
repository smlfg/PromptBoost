# PromptBoost v0.1.1

Local prompt compiler/eval logger for personal prompt work.

PromptBoost reads PromptGarage examples read-only, accepts a raw prompt, selects
task/harness/model rules, generates a deterministic boosted prompt, runs a
local no-new-facts check, and stores boost runs in `promptboost.db`.

## Run

```bash
uv run uvicorn promptboost.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## Agentic mode

The deterministic `/boost` route remains the default. The optional `/boost/agentic`
route runs an OpenAI-compatible LLM prepass first, validates the JSON task draft,
then falls back to deterministic PromptBoost if the LLM is unavailable or adds
suspected new facts.

```bash
export PROMPTBOOST_LLM_API_KEY=...
export PROMPTBOOST_LLM_BASE_URL=https://api.openai.com/v1
export PROMPTBOOST_LLM_MODEL=gpt-5.4-mini
```

## Test

```bash
uv run pytest
```

## Notes

- PromptGarage is read with SQLite immutable/read-only connections.
- No paid LLM APIs are called in v0.1.
- The no-new-facts check is conservative and deterministic; it is a warning gate,
  not a semantic proof system.
