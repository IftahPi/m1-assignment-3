# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course assignment (Nebius Academy, "From AI Model to AI Agent", Assignment 3): a **customer-service data-analyst agent** over the Bitext Customer Service dataset (26,872 rows). It answers structured questions ("how many refund requests?"), open-ended ones ("summarize the FEEDBACK category"), and politely declines out-of-scope questions. Built as a LangGraph routed ReAct agent; tools are also exposed over MCP (later task).

`IMPLEMENTATION_PLAN.md` is the authoritative build plan and `DELIVERABLES_NOTES.md` is the per-task grading checklist + final packing step. Read both before non-trivial work.

## Commands

Always use the project venv (Python 3.11): `.venv/bin/python`.

```bash
.venv/bin/python -m pytest -q                       # full unit suite (LLM mocked; fast, free, deterministic)
.venv/bin/python -m pytest tests/test_graph.py -q   # one test file
.venv/bin/python -m pytest tests/test_graph.py::test_repeated_tool_call_triggers_force_answer  # one test
.venv/bin/python main.py                            # interactive CLI agent (needs .env)
.venv/bin/python router_eval.py                     # LIVE router accuracy harness (hits Nebius)
.venv/bin/python agent_eval.py                      # LIVE end-to-end agent harness (hits Nebius)
.venv/bin/python smoke_test_nebius.py               # LIVE connectivity + model-list check
```

Requires a `.env` (gitignored) with `NEBIUS_API_KEY=...` on a single line. Copy `.env.example`.

## Architecture (the big picture)

**Dual-model strategy** (`nebius_client.py`): all LLM calls go through `make_llm(model, temperature)` → `ChatOpenAI` pointed at Nebius's OpenAI-compatible endpoint. A small cheap MoE (`ROUTER_MODEL` = Qwen3-30B-A3B-Instruct) classifies; a ReAct-native model (`GENERATOR_MODEL` = openai/gpt-oss-120b) reasons/answers — chosen over Llama-3.3-70B after an A/B on the rubric (gpt-oss-120b emits visible Thought content alongside tool_calls; Llama's function-calling mode keeps content empty). Only Nebius Token Factory models are allowed.

**Layered, DRY data access** — the key design decision:
- `dataset/analytics.py` holds **pure functions** (DataFrame in → typed JSON-serializable value out): `list_categories`, `list_intents`, `count_records`, `get_examples`, `intent_distribution`, `search_examples`. No LangChain, no I/O. Filters: category/intent case-insensitive, `text_contains` substring over the customer `instruction`.
- `agent/tools.py` wraps each pure function as a LangChain `@tool` with a Pydantic input schema (`agent/schemas.py`) and a description that tells the LLM *when* to use it.
- The same pure functions will back the FastMCP server (`mcp_server.py`, later task). Change behavior in `analytics.py` once; tools and MCP both follow.
- `dataset/loader.py` loads/caches the CSV (downloads from Hugging Face if `data/` is absent), memoized with `lru_cache`.

**The graph** (`agent/graph.py`, state in `agent/state.py`): a routed ReAct loop, NOT `create_react_agent` (we need a router gate first):
```
START → router → (out_of_scope → decline) | (else → agent)
agent ⇄ tools                 # ReAct loop while the LLM requests tools
agent → force_answer          # repeated identical tool call (no progress) → answer with no tools
agent → fallback              # iteration budget (MAX_ITERATIONS=12) exhausted
decline / force_answer / fallback / plain answer → END
```
- `router` reuses `agent/router.py::classify_query` (structured output → `RouteDecision`); its prompt enumerates all 11 categories + 27 intents and the hard "never answer out-of-scope from general knowledge" rule.
- Robustness guard: `_route_after_agent` computes a `name+sorted-args` signature per tool call; if all requested calls were already executed it diverts to `force_answer` (generator invoked with **no tools**, redundant tool-call message stripped via `RemoveMessage` so history stays valid for checkpoint replay). The generator runs at `temperature=0.2` to avoid deterministic loops.

**CLI** (`cli/repl.py`, thin `main.py`): streams the graph (`stream_mode="updates"`) and renders each step — router decision, tool calls, observations, then the answer — not just the final answer. `run_repl` takes injected `graph`/`input_fn`/`output_fn` so it's testable without an LLM.

## Conventions (binding — from the `python-oop` skill)

- **TDD, Red→Green→Refactor.** No production code without a failing test first.
- **Test the boundary, not internals.** Pure `analytics.py` is tested directly against a fixture DataFrame (no mocks). The **only** thing mocked is the Nebius LLM call (`make_llm` / `classify_query`). LLM *accuracy* is checked by the live `*_eval.py` scripts, which are deliberately NOT pytest (they cost tokens and are non-deterministic).
- Type hints everywhere; `str | None` (never `Optional`); no `from __future__ import annotations` (3.11). `Literal` for enums.
- Absolute imports from the repo root (`from agent.graph import build_graph`); no `sys.path` hacks. Root-level packages so `python main.py` runs with no install. `main.py` is wiring only.
- ≤200 lines/file; docstrings on public functions; private helpers prefixed `_`.

## Status

Task 1 (router + tools + ReAct loop + CLI) is complete and committed through step 3; the agent/tools/graph (step 4) is built and live-verified but commit status is tracked in the agent memory. Still to do: Task 2 (SQLite checkpoint memory + `--session` + user profile), Task 3 (FastMCP server), `README.md`, and the deliverable zip. See `IMPLEMENTATION_PLAN.md` §11 for the TDD build order and acceptance gates.
