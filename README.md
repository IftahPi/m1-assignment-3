# Customer Service Data Analyst Agent

A LangGraph ReAct agent that answers questions about the **Bitext Customer Service** dataset (26,872 customer-support messages paired with agent replies, across 11 categories and 27 intents). It handles **structured** questions ("how many refund requests?"), **unstructured** ones ("summarize how agents respond to complaints"), and **politely declines out-of-scope** questions instead of answering them from world knowledge. Tools are exposed via FastMCP.

> **Course:** Nebius Academy — *From AI Model to AI Product*, Module 1, Assignment 3.
> **LLM provider:** **Nebius Token Factory only** (assignment requirement).

---

## Setup (≤ 5 minutes)

Requires **Python 3.11**, `git`, and a **Nebius Token Factory API key** ([studio.nebius.com](https://studio.nebius.com/)).

```bash
git clone https://github.com/IftahPi/m1-assignment-3.git
cd m1-assignment-3

# 1. Virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. API key — copy the example and paste your key
cp .env.example .env
# then edit .env and put your real key on the NEBIUS_API_KEY line (single line, no quotes)

# 4. Smoke-test the Nebius connection
python smoke_test_nebius.py
```

The smoke test lists the Llama models your account can see and pings both configured models. The first run of the CLI will also download the dataset CSV (~19 MB) from Hugging Face into `data/` and cache it.

---

## Running the CLI

```bash
python main.py
```

You'll get a welcome banner; then type questions at the `You> ` prompt. Type `quit` or `exit` to leave. Blank input re-prompts. The CLI streams the **router decision, every tool call, every observation, and the final answer** — not just the answer.

### Example queries to try

**Structured** (concrete, data-driven answers):
```
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples from the SHIPPING category.
What is the distribution of intents in the ACCOUNT category?
Show me examples of people wanting their money back.
How many complaints and how many refunds? What is the total of the two?
```

**Unstructured** (open-ended summarization):
```
Summarize the FEEDBACK category.
Summarize how agents respond to complaints.
How do customer service representatives typically respond to cancellation requests?
```

**Out-of-scope** (should be politely declined):
```
Who is the president of France?
Write me a poem about customer service.
What's the best CRM software for handling complaints?
```

### Example trace (structured query)
```
You> How many refund requests did we get?
  🧭 router → structured
  🔧 count_records(category='REFUND')
  📊 2992

🤖 We received 2992 refund-related requests (category 'REFUND').
```

### Example trace (multi-step + arithmetic)
```
You> How many complaints and how many refunds? Total?
  🧭 router → structured
  🔧 count_records(intent='complaint')
  📊 1000
  🔧 count_records(category='REFUND')
  📊 2992

🤖 Total: 1000 + 2992 = 3992.
```

---

## Architecture

### The graph

A routed ReAct loop, built as a custom LangGraph `StateGraph` (not the off-the-shelf `create_react_agent` — we need a router gate up front).

```
   START
     │
     ▼
  ┌────────┐    structured output → RouteDecision
  │ router │    (small/cheap model: Qwen3-30B-A3B-Instruct)
  └───┬────┘
      │
      ├── out_of_scope ──► decline  ──►  END   (fixed polite refusal,
      │                                         never calls world knowledge)
      ▼
   structured / unstructured
      │
      ▼
  ┌────────┐ ◄────────────────────────────────────────────────────────┐
  │ agent  │  generator (Llama-3.3-70B-Instruct) bound to tools       │
  └───┬────┘                                                          │
      │  AIMessage(content?, tool_calls?)                             │
      ▼                                                               │
   ┌─────────────────┐                                                │
   │  tool_calls?    │                                                │
   └──────┬──────────┘                                                │
          │                                                           │
          ├── no  ────────────────────────────────►  END              │
          │                                                           │
          ├── yes & iter ≥ MAX_ITERATIONS (=12) ──► fallback ──► END  │
          │                                                           │
          ├── yes & all requested calls already executed ──►          │
          │     force_answer (LLM without tools)        ──► END       │
          │                                                           │
          └── yes & new call ──►  tools  ─────────────────────────────┘
                                  (runs the @tool, appends ToolMessage)
```

**Key node responsibilities:**

| Node | Job |
|---|---|
| `router` | Classifies the query into `structured` / `unstructured` / `out_of_scope`. Uses **structured output** (Pydantic `RouteDecision`) so it never returns free text. Its prompt enumerates all 11 categories and all 27 intents. |
| `decline` | Polite fixed reply for out-of-scope queries. **Never** calls the generator or world knowledge — the requirement. |
| `agent` | The generator (with `bind_tools`). Reads the conversation, decides which tool to call (or to answer). Runs at `temperature=0.2`. |
| `tools` | LangGraph `ToolNode` executes the requested tool against the cached DataFrame, appends a `ToolMessage`. |
| `force_answer` | **Anti-loop guard.** If the model repeats an identical tool call (`name + sorted args`), this node invokes the generator *without tools* and asks it to answer from the data already collected — and strips the redundant tool-call message via `RemoveMessage` so history remains valid for Task 2 checkpoint replay. |
| `fallback` | Graceful "I couldn't work that out in my step budget" message when `iterations ≥ MAX_ITERATIONS`. |

### Model choice & justification

A **dual-model strategy** — a small cheap model for the routing decision, a larger ReAct-native model for tool-calling, reasoning, and summarization. This is the W4 "Routing" workflow pattern applied to the agent itself.

| Role | Model | Why |
|---|---|---|
| **Router** | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE with **~3B active parameters** — very cheap and fast per call. The router only needs to classify, not reason. Verified that `.with_structured_output(RouteDecision)` returns clean enum values. The "Instruct" (non-thinking) variant avoids slow chain-of-thought outputs on a classification task. **Live router accuracy: 13/13 on the assignment's example queries** (`router_eval.py`). |
| **Generator** | `openai/gpt-oss-120b` | Chosen over `meta-llama/Llama-3.3-70B-Instruct` after a head-to-head A/B against the agent rubric (see `agent_eval.py`). Three concrete wins: (1) it emits a brief **reasoning sentence as message content alongside its tool calls**, so the CLI shows the actual `💭 THOUGHT` line the assignment's "print reasoning steps" requirement asks for — Llama-3.3-70B in function-calling mode keeps that content empty; (2) cleaner tool-call discipline (no junk null-string args, no redundant duplicates triggering `force_answer`, no fabricated example rows when a tool returns empty); (3) noticeably better answer formatting (markdown tables for examples and distributions). Run at `temperature=0.2` to break deterministic loops without losing reproducibility. |

Both go through `nebius_client.make_llm()`, which wraps `langchain_openai.ChatOpenAI` with `base_url=https://api.studio.nebius.com/v1/`.

> **A/B summary** — same 11-query trace dump, identical router, only the generator swapped. Llama-3.3-70B: 8 PASS / 1 partial / **2 hard fails** (one multi-step regression, one fabricated-examples answer on an empty tool result). gpt-oss-120b: **11/11 PASS**, with visible `💭 THOUGHT` lines on 6 of the 11 cases. The 120B is larger and slightly slower / costlier per call, but the quality gap on this assignment justifies it.

### Tools

A core design choice: the data-access functions in `dataset/analytics.py` are **pure** (DataFrame in → JSON-serializable value out, no LangChain or LLM dependency). The agent wraps them with `@tool` and a Pydantic schema; the MCP server reuses the same functions. One source of truth.

| Tool | Input schema | Returns | Use for |
|---|---|---|---|
| `list_categories` | — | `list[str]` | "What categories exist?" |
| `list_intents` | `category? : <one of 11>` | `list[str]` | Listing intents inside a category |
| `count_records` | `category? · intent? · text_contains?` | `int` | All "how many…" questions; filters AND-combine |
| `get_examples` | `category? · intent? · text_contains? · n` | `list[dict]` | "Show me examples of…" |
| `intent_distribution` | `category?` | `dict[str, int]` | "What is the distribution of intents in X?" |
| `search_examples` | `query · n` | `list[dict]` | When the user describes something in their own words ("money back") |

**Schema robustness:** `category` and `intent` are typed as Pydantic `Literal` enums of the dataset's actual values, with a `BeforeValidator` that case-normalises ("shipping" → "SHIPPING"). So the LLM **cannot** put a category value into the intent slot — if it tries, Pydantic returns a validation error and the model retries with the right slot.

### Memory

Two distinct kinds, mirroring the CoALA taxonomy from the lectures:

- **Episodic** (Task 2a) — the literal conversation. Persisted via a `SqliteSaver` checkpointer keyed by `thread_id = --session`. Same `--session` on a later run resumes the prior conversation; follow-ups like *"what about refunds?"* and *"what is the total of the two?"* inherit prior context because the router is given a short "Recent conversation:" block before classifying.
- **Semantic** (Task 2b) — a per-user **freeform Markdown profile** at `profiles/<user_id>.md` capturing distilled facts (name, recurring interests, preferences) — **not** a transcript replay. Updated by a `summary` step in `main.py` that runs at session end: pull the final state from the checkpointer, ask the generator LLM to merge the new session against the prior profile, write the result. The summary prompt has an explicit **contradiction rule** — when a new fact contradicts an existing one, the old fact is replaced (not kept alongside the new one).

Personal questions get their own graph path — the router has **four labels** (`structured` / `unstructured` / `out_of_scope` / `personal`) and routes *"what do you remember about me?"*-style queries to a dedicated `personal` node. The personal node:

- runs on the small **router model** (Qwen3-30B-A3B-Instruct) — paraphrasing a tiny profile file does not need the 70-120B generator;
- binds **exactly one tool**, a closure-scoped `get_personal_info(user_id)` locked to the current session's user — the LLM cannot read another user's profile even if it tries;
- has **no access to the data tools** at all — fabrication from dataset facts is structurally impossible;
- if the profile is empty, the prompt requires the model to say *"I don't have that on file yet"* rather than guess.

The path scheme — *where on disk a user's profile lives* — is owned by one pure function (`agent/profile.py::get_personal_storage_file`); every reader and writer (the tool, the summary code, tests) goes through it, so changing the layout later is a one-line edit.

CLI usage:

```bash
python main.py --user alice --session s1   # Alice's first chat
# … Alice tells the agent about her interests, then quits
# → 💾 Updated profile for user 'alice' saved to profiles/alice.md

python main.py --user alice --session s2   # new conversation, same user
You> What do you remember about me?
  🧭 router → personal
  🔧 get_personal_info(user_id='alice')
  📊 (the alice.md contents)
🤖 Your name is Alice.
```

`--user` defaults to the literal `"default"` (independent of `--session`) — so one user can have many sessions sharing the same profile. Omitting both gives you a "default" user under a "default" session.

#### Planned but not yet implemented

- **Timestamping facts.** Every line in `profiles/<user_id>.md` will eventually be prefixed with the date+time it was added, so old facts can be aged out and the summary node can decide which to keep based on recency. The schema is intentionally freeform Markdown today so this change can be layered on without breaking the storage contract.

---

## MCP server

> **Status:** Task 3 in progress.

The MCP server exposes (at least) `list_categories`, `count_records`, `get_examples`, and `intent_distribution` as MCP tools — wrapping the **same** `dataset/analytics.py` functions the agent uses.

**Starting the server (once implemented):**
```bash
python mcp_server.py
```

**Connecting a client and calling a tool:**
```python
import asyncio
from fastmcp import Client

async def main():
    # Connect to the server over stdio (auto-spawns the script).
    async with Client("mcp_server.py") as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        # Count refund-related rows.
        result = await client.call_tool("count_records", {"category": "REFUND"})
        print("Refund records:", result)

        # Fetch a few SHIPPING examples.
        examples = await client.call_tool("get_examples", {"category": "SHIPPING", "n": 3})
        print("Shipping examples:", examples)

asyncio.run(main())
```

The server is a `FastMCP("bitext-data-analyst")` instance with `@mcp.tool`-decorated wrappers that share the analytics layer; no duplicated logic.

---

## Testing & evaluation

The unit suite mocks the only true system boundary — the Nebius LLM HTTP call — so it runs offline, fast, and free. LLM **accuracy** is checked by separate, opt-in live harnesses that make real API calls.

```bash
python -m pytest -q           # 52 tests, LLM mocked, all green
python router_eval.py         # live router accuracy (13 assignment queries)
python agent_eval.py          # live end-to-end agent (9 outcome checks)
```

Last live runs: `router_eval` 13/13, `agent_eval` 9/9.

---

## Repository layout

```
m1-assignment-3/
  nebius_client.py            # Nebius config + make_llm()
  dataset/
    loader.py                 # caches the CSV (downloads on first run)
    analytics.py              # PURE data functions — reused by tools AND MCP
  agent/
    schemas.py                # Pydantic models incl. category/intent Literals
    router.py                 # classify_query() with structured output
    tools.py                  # @tool wrappers around dataset.analytics
    state.py                  # AgentState TypedDict
    graph.py                  # the StateGraph above
    profile.py                # per-user semantic memory (Task 2b)        [planned]
  cli/
    repl.py                   # interactive loop + reasoning renderer
  mcp_server.py               # FastMCP server (Task 3)                   [planned]
  main.py                     # thin entry point
  tests/                      # 52 unit tests (LLM boundary mocked)
  router_eval.py              # live router accuracy harness
  agent_eval.py               # live end-to-end agent harness
  IMPLEMENTATION_PLAN.md      # the build plan (STATUS section at top)
  DELIVERABLES_NOTES.md       # per-task grading checklist
  CLAUDE.md                   # codebase guide for future LLM sessions
```

`data/` (cached CSV), `.venv/`, `.env`, `*.sqlite`, and `profiles/` are gitignored.

---

## Submission notes

- **No external LLMs**: every LLM call goes through `nebius_client.make_llm()` against the Nebius Token Factory endpoint.
- **Dataset**: Bitext Customer Service ([Hugging Face](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)). Cached to `data/` on first run; not committed.
