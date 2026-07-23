# PromptBoost documentation

Short index for the PromptBoost v0.1.1 local prompt compiler.

## Guides

| Document | Description |
|----------|-------------|
| [../README.md](../README.md) | Main project README — quick start, features, API |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Pipeline stages, components, and data flow |

## Visuals

| Image | Description |
|-------|-------------|
| [images/promptboost-pipeline.png](images/promptboost-pipeline.png) | End-to-end architecture flow |
| [images/promptboost-ui-mock.png](images/promptboost-ui-mock.png) | Web UI layout overview |

## Key concepts

- **Raw task boundary** — The user's original prompt remains the source of truth in every boosted candidate.
- **Harness policy** — Runtime rules for a target agent environment (tools, verification gates, failure modes).
- **Model adapter** — Model-family-specific prompt shaping rules.
- **No-new-facts guard** — Deterministic check that the boosted prompt did not introduce unsupported domain claims.
- **Agentic prepass** — Optional LLM step that extracts a structured task draft before deterministic compilation.
