# Implementation Plan — Customer Service Data Analyst Agent

**Audience:** an implementing model/developer. Follow this top-to-bottom. Do not invent extra scope.
**Stack is already installed** in `.venv` (Python 3.11). See `requirements.txt`. Versions are **langgraph 1.x / langchain 1.x** — use current (post-1.0) import paths, not old tutorials.

---

## STATUS — resume here (updated as we go)

**DONE & committed (steps 1–3):** planning docs · pinned `requirements.txt` · `.gitignore` · `nebius_client.py` (+ smoke test, connectivity verified) · query router (`agent/router.py`, `agent/schemas.py::RouteDecision`) · CLI shell with welcome banner (`cli/repl.py` step-3 form) · `router_eval.py` (13/13 live).

**DONE, NOT yet committed (step 4 = Task 1 complete):** `dataset/{loader,analytics}.py` · tool input schemas + `agent/tools.py` (6 tools) · `agent/state.py` · `agent/graph.py` (routed ReAct loop + `force_answer` dedupe guard + `fallback`) · `cli/repl.py` rewritten to stream reasoning · `agent_eval.py` (8/8 live). **42 unit tests pass.** Generator runs at `temperature=0.2`; `MAX_ITERATIONS=12`. Self-ranked 9/10.

**TODO (in order):**
1. Commit step 4.
2. **Task 2a** — SQLite checkpointer (`langgraph-checkpoint-sqlite`), `main.py --session <id>`, persistence across restart, follow-up queries. (`build_graph(checkpointer=...)` already accepts a checkpointer.)
3. **Task 2b** — per-user profile (distilled facts, persisted, injected into the agent system prompt; "what do you remember about me?"). See plan §7.
4. **Task 3** — `mcp_server.py` (FastMCP) exposing ≥3 `dataset.analytics` functions; README client snippet.
5. `README.md` (5-min clone-to-run, architecture, model choice, MCP connect) and the deliverable **zip** (see `DELIVERABLES_NOTES.md`).
6. Optional bonuses (Streamlit UI; query recommender).

Architecture overview for fast onboarding is in `CLAUDE.md`. Detailed per-file specs below.

---

## CODING STANDARDS (binding — from the `python-oop` skill)

These override any convention implied elsewhere in this doc. Non-negotiable:

- **TDD, Red→Green→Refactor.** No production code without a failing test first. Write one minimal failing pytest, watch it fail for the right reason, then write the simplest code to pass. Tests live in `tests/`, plain pytest functions (no `unittest.TestCase`).
  - The `data_tools` pure functions are tested **directly, no mocking** (they're pure functions over a small fixture DataFrame).
  - Mock **only** the true system boundary = the Nebius LLM HTTP call. Never mock our own graph/tool internals.
  - Use `tmp_path` for the SQLite checkpoint and profile-file tests.
- **Typing everywhere.** Annotate every param + return. Use `str | None` (NOT `Optional[str]`), `Literal[...]` for enums, parameterised generics (`list[dict]`, `dict[str, int]`). Do **not** add `from __future__ import annotations` (3.11 doesn't need it). Avoid `Any` except at the raw-API boundary.
- **Package layout** (see below): `__init__.py` in each package, **absolute imports only** (`from agent.graph import build_graph`), never `sys.path.insert`, never relative imports, top-level imports only.
- **Thin entry point.** `main.py` does wiring only (argparse → build graph → start REPL). The REPL/printing logic lives in its own module so it's importable + testable.
- **Single Responsibility, small units.** ≤ 200 lines/file (300 hard ceiling), functions ≤ ~20 lines, one job each. Guard clauses over nesting.
- **Naming.** `snake_case` funcs/modules, `PascalCase` classes, `UPPER_SNAKE` constants, `is_/has_` booleans, spell words out. No `utils.py`/`helpers.py`/`common.py`. Private helpers get a `_` prefix; public surface stays minimal.
- **Data modeling.** `@dataclass` (or Pydantic where a schema is needed) over raw dicts for structured records; `default_factory` for mutable defaults; validate invariants in `__post_init__`.
- **Docstrings** on every public function/class.
- **No `print()` for debugging**; the CLI's user-facing reasoning output is a deliberate feature (use a small render function), not stray prints. Swallow no exceptions silently.

### Project layout (root-level packages — `python main.py` works with no install/path hacks)
```
m1-assignment-3/
  data/bitext_customer_support.csv   # cached (gitignored)
  nebius_client.py           # Nebius settings + make_llm()  (root module)
  dataset/
    __init__.py
    loader.py                # load_dataframe() (cached)
    analytics.py             # PURE functions: count/examples/distribution/search  (was "data_tools")
  agent/
    __init__.py
    schemas.py               # Pydantic input schemas + RouteDecision
    tools.py                 # @tool wrappers around dataset.analytics  (was "agent_tools")
    state.py                 # AgentState TypedDict
    router.py                # router node (RouteDecision via structured output)
    graph.py                 # StateGraph: router/decline/agent/tools/fallback nodes
    profile.py               # user-profile load/update/persist (Task 2b)
  cli/
    __init__.py
    repl.py                  # interactive loop + reasoning renderer (importable)
  mcp_server.py              # FastMCP exposing >=3 analytics functions (Task 3)
  main.py                    # THIN: argparse + wire + repl.run()
  conftest.py                # makes root packages importable under pytest
  tests/
    test_analytics.py        # pure-function tests (no mocks)
    test_tools.py
    test_router.py           # mocks the LLM boundary
    test_graph.py
    test_profile.py
  README.md  requirements.txt  .env.example  .gitignore
```
- **Imports are absolute from the repo root**: `from nebius_client import make_llm`, `from dataset.analytics import count_records`, `from agent.graph import build_graph`. Running `python main.py` from the root puts the root on `sys.path`; `conftest.py` does the same for pytest. No `src/` layer, no `pip install -e .`, no `sys.path.insert`.
- (Rename note: earlier sections say `data_tools.py`/`agent_tools.py` — real homes are `dataset/analytics.py` and `agent/tools.py`. Keep the function names.)

---

## 0. Ground truth: the dataset

Loaded from Hugging Face: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`,
file `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`.

- **26,872 rows**, columns: `flags`, `instruction` (customer text), `category`, `intent`, `response` (agent reply).
- **11 categories:** `ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION`
- **27 intents:** cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order, track_refund
- Mapping hints for natural language: "refund requests" → category `REFUND` (or intent `get_refund`); "complaints" → intent `complaint`; "money back" → search instruction text. Text contains placeholders like `{{Order Number}}` — leave them as-is when showing examples. ~48% of responses contain such placeholders.

- **category → intents map** (put this in the agent's system prompt so it can map NL → filters):
  ```
  ACCOUNT      → create_account, delete_account, edit_account, recover_password, registration_problems, switch_account
  ORDER        → cancel_order, change_order, place_order, track_order
  REFUND       → check_refund_policy, get_refund, track_refund
  FEEDBACK     → complaint, review
  PAYMENT      → check_payment_methods, payment_issue
  INVOICE      → check_invoice, get_invoice
  DELIVERY     → delivery_options, delivery_period
  SHIPPING     → change_shipping_address, set_up_shipping_address
  CONTACT      → contact_customer_service, contact_human_agent
  CANCEL       → check_cancellation_fee
  SUBSCRIPTION → newsletter_subscription
  ```
- Row counts are uneven (ACCOUNT 5986 … CANCEL 950) — don't assume uniform sampling.
- `flags` = linguistic-variation tags on the instruction (B basic, I interrogative, C coordinated, N negation, M morphological, L semantic, P politeness, Q colloquial, W offensive, K keyword, E abbreviation, Z typos). Not needed by the agent; explains why instructions are noisy → favor substring search over exact match.

---

## 1. Nebius Token Factory config (`nebius_client.py`)

Nebius exposes an **OpenAI-compatible** API, so use `langchain_openai.ChatOpenAI` with a custom base URL.

- Base URL: `https://api.studio.nebius.com/v1/` (Nebius AI Studio / Token Factory).
- API key from env `NEBIUS_API_KEY`.
- **Recommended models (CONFIRM with user before coding):**
  - `ROUTER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"` — cheap/fast classifier.
  - `GENERATOR_MODEL = "openai/gpt-oss-120b"` — ReAct-native (emits Thought content alongside tool_calls); chosen over Llama-3.3-70B after A/B on the rubric.
  - (If user picks single-model: use the 70B for both.)
- Provide a helper:
  ```python
  def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
      return ChatOpenAI(model=model, temperature=temperature,
                        base_url=os.environ["NEBIUS_BASE_URL"],
                        api_key=os.environ["NEBIUS_API_KEY"])
  ```
- `.env.example`: `NEBIUS_API_KEY=` and `NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/`. Load with `python-dotenv`.

> ⚠️ Verify the exact Nebius model IDs and base URL against current Nebius docs before finalizing — IDs change.

---

## 2. Data layer (`data_loader.py`)

- `load_dataframe() -> pandas.DataFrame`: download the CSV once (HF `hf://...` path or `huggingface_hub.hf_hub_download` + `pd.read_csv`), cache in a module-level global so it loads once per process. Return the DataFrame.
- Optionally cache to a local `./data/bitext.csv` to avoid re-downloading.

---

## 3. Pure data functions (`data_tools.py`) — the DRY core

These are plain typed functions on the DataFrame, **no LangChain/MCP imports**. Reused by both the agent and the MCP server. Filters are case-insensitive on category/intent.

Implement (names matter — they become tool names):
1. `list_categories() -> list[str]` — all distinct categories.
2. `list_intents(category: str | None = None) -> list[str]` — intents, optionally scoped to a category.
3. `count_records(category: str | None = None, intent: str | None = None, text_contains: str | None = None) -> int` — count rows matching the filters (AND-combined).
4. `get_examples(category: str | None = None, intent: str | None = None, text_contains: str | None = None, n: int = 5) -> list[dict]` — up to `n` sample rows, each `{category, intent, instruction, response}`.
5. `intent_distribution(category: str | None = None) -> dict[str, int]` — intent → count, optionally within a category.
6. `search_examples(query: str, n: int = 5) -> list[dict]` — case-insensitive substring search over `instruction` (handles "people wanting their money back").

Each returns typed, JSON-serializable data. Validate category/intent against the known sets; on unknown value return an empty result (the agent will react).

---

## 4. Schemas (`schemas.py`)

Pydantic v2 models (use `str | None`, not `Optional[str]`):
- One input schema per tool that has params, e.g. `CountRecordsInput(category: str | None, intent: str | None, text_contains: str | None)`, `GetExamplesInput(... , n: int = 5)`, `IntentDistributionInput(category: str | None)`, `SearchExamplesInput(query: str, n: int = 5)`, `ListIntentsInput(category: str | None)`.
- `RouteDecision(route: Literal["structured","unstructured","out_of_scope"], reason: str)` for the router's structured output.
- Give every field a `Field(description=...)` — these descriptions are graded.

---

## 5. Agent tools (`agent_tools.py`)

Wrap each `data_tools` function as a LangChain tool using `@tool(args_schema=...)` (from `langchain_core.tools`). 
- **Write a strong docstring/description for each** — a human (and the LLM) must know *when* to use it. (e.g. `count_records`: "Return how many dataset rows match the given category/intent/text filter. Use for 'how many X' questions.")
- Return the typed data from the pure function (LangChain will serialize).
- Export a `TOOLS = [...]` list.

There is **no separate "summarize" tool**: for unstructured questions the agent calls `get_examples`/`search_examples` to pull representative `instruction`+`response` rows, then the LLM summarizes them itself. (Teach: tools = deterministic data ops; LLM = reasoning.)

---

## 6. The graph (`agent.py`) — custom StateGraph

State (TypedDict): `messages: Annotated[list, add_messages]`, `route: str`, `iterations: int`, plus profile fields if needed.

Nodes & edges:
1. **router** — uses ROUTER_MODEL with `.with_structured_output(RouteDecision)` on the latest user message (+ short context). Sets `state["route"]`.
2. Conditional edge from router:
   - `out_of_scope` → **decline** node → END. Decline node returns a fixed polite message ("I can only answer questions about the Bitext customer-service dataset…"). **Do not call the LLM's world knowledge.**
   - else → **agent** node.
3. **agent** — GENERATOR_MODEL bound to `TOOLS` (`llm.bind_tools(TOOLS)`). Appends its AIMessage. Increments `state["iterations"]`.
4. Conditional edge from agent:
   - if last AIMessage has tool_calls AND `iterations < MAX_ITERATIONS` → **tools** node (`ToolNode(TOOLS)` from `langgraph.prebuilt`) → back to **agent**.
   - if `iterations >= MAX_ITERATIONS` → **fallback** node (returns graceful "couldn't complete in time" message) → END.
   - else (no tool calls) → END.
- `MAX_ITERATIONS = 12`.
- System prompt for the agent: explains it's a data analyst over the Bitext dataset, lists categories/intents, instructs it to use tools for facts and to summarize from fetched rows for open-ended questions, and to inject the user profile (Task 2b).

**Checkpointer:** compile with `SqliteSaver`:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
# context-manager or .from_conn_string("checkpoints.sqlite")
graph = builder.compile(checkpointer=saver)
```
Invoke with `config={"configurable": {"thread_id": session_id}, "recursion_limit": 50}`.

---

## 7. User profile (`profile_store.py`) — Task 2b

- Store per user at `profiles/<user>.json` (or `.md`). Functions: `load_profile(user) -> dict`, `save_profile(user, dict)`, `update_profile(user, conversation_snippet) -> dict`.
- `update_profile` calls an LLM with the recent turn(s) and the current profile, asking it to return updated distilled facts (name, recurring topics, preferences) — JSON. Merge & persist.
- Wire either as a **profile node** that runs after each agent turn (distill + save), or call `update_profile` from the CLI loop after each exchange. Inject `load_profile(user)` text into the agent's system prompt so "What do you remember about me?" is answerable.
- Profile is keyed by `--user` (default it to the session id if `--user` omitted).

---

## 8. CLI (`main.py`) — Task 1 CLI + Task 2a session

- `argparse`: `--session` (default `"default"`), `--user` (default = session).
- Build/compile the graph with SqliteSaver.
- Interactive `while True:` loop: read input (`quit`/`exit` to stop).
- Use `graph.stream(input, config, stream_mode="updates")` and **print each step**: when an AIMessage has `tool_calls`, print `🔧 calling <tool>(<args>)`; when a ToolMessage arrives, print `📊 observation: <result>`; finally print the assistant's answer. This satisfies "print reasoning steps."
- After each turn, update the user profile.
- Restarting with the same `--session` restores history (checkpointer does this automatically via thread_id).

---

## 9. MCP server (`mcp_server.py`) — Task 3

- `from fastmcp import FastMCP`; `mcp = FastMCP("bitext-data-analyst")`.
- Register **≥3** tools with `@mcp.tool` wrapping the **same `data_tools` functions** (e.g. `count_records`, `get_examples`, `intent_distribution`, `list_categories`). Keep typed signatures + docstrings.
- `if __name__ == "__main__": mcp.run()` (stdio transport by default; or `transport="http"` if README documents it).
- README must include a **client snippet**, e.g. using `fastmcp.Client`:
  ```python
  import asyncio
  from fastmcp import Client
  async def main():
      async with Client("mcp_server.py") as c:
          print(await c.call_tool("count_records", {"category": "REFUND"}))
  asyncio.run(main())
  ```
  Verify against installed `fastmcp` 3.x API before finalizing.

---

## 10. README.md (graded — 5-min clone-to-run)

Sections: (1) overview, (2) setup (`python -m venv`, `pip install -r requirements.txt`, copy `.env.example`→`.env`, add Nebius key), (3) run the CLI (`python main.py --session demo`), (4) example queries + expected behavior incl. out-of-scope decline, (5) MCP: start server + client snippet, (6) architecture: graph diagram (router→decline/agent⇄tools→fallback), tool list with one-line descriptions, **model choice + justification** (router vs generator), memory design (SqliteSaver + profile file).

---

## 11. Build order & acceptance gates (TDD)

Each step is **Red→Green→Refactor**: write the failing test(s) first, watch them fail, then implement.

1. `nebius_client.py` + `.env.example` → smoke-test: one real Nebius call returns text. (boundary; verify manually once, not a unit test.)
2. `src/dataset/loader.py` + `src/dataset/analytics.py` → **`tests/test_analytics.py` first** against a tiny fixture DataFrame: counts, filters (category/intent/text), examples, distribution, search, unknown-value → empty. No mocks.
3. `src/agent/schemas.py` + `src/agent/tools.py` → `tests/test_tools.py`: tool wrappers call through and return typed data; schemas validate/reject bad input.
4. `src/agent/state.py` + `src/agent/graph.py` (no memory yet) + `src/cli/repl.py` + thin `main.py` → `tests/test_router.py` (mock LLM boundary: each label routes correctly, OOS → decline) and `tests/test_graph.py` (multi-step chains tools; max-iter → fallback). **Task 1 acceptance:** all 8 demo queries behave (OOS decline, multi-step, reasoning printed, fallback).
5. Add SqliteSaver + `--session` → `tests/test_graph.py` persistence case with `tmp_path`. **Task 2a acceptance:** quit & restart, history restored, follow-ups work.
6. `src/agent/profile.py` → `tests/test_profile.py` (`tmp_path`; mock the distill LLM call). **Task 2b acceptance:** "what do you remember about me?" after stating a name/preference; persists across restart.
7. `src/mcp_server.py` → **Task 3 acceptance:** client snippet returns a real result.
8. README. Then (optional) bonuses. Then zip per DELIVERABLES_NOTES.md.

**After each numbered step, run the full suite (must be green) and verify the acceptance check before moving on.**

---

## Open decision for the user
- **Model strategy** (dual small-router+large-generator vs single 70B) — user deferred; confirm before step 1. Plan defaults to dual.
- Whether to attempt **Bonus A/B** — deferred until 100 core pts are solid.
