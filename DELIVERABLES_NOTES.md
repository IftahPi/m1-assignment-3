# Assignment 3 — Deliverables Checklist & Grading Notes

> Due **2026-05-29**. Submit a GitHub repo (or zip).
> Only **Nebius Token Factory** models allowed for any LLM call.

## What gets submitted (the package)
- [ ] `requirements.txt` (pinned versions) — ✅ already written
- [ ] `README.md` — clone→running agent in **5 minutes** by README alone. Must cover:
      setup steps · how to run the CLI · how to connect an MCP client to a tool · architecture overview (which Nebius model(s) & why, what tools defined)
- [ ] `.env.example` (Nebius API key var; never commit the real key)
- [ ] All source files (see IMPLEMENTATION_PLAN.md)
- [ ] The SQLite checkpoint file is generated at runtime — do **not** need to ship it (but a demo session is nice)

---

## Task 1 — Initial Agent (50 pts)
Graded sub-parts → deliverables that earn each:

| Pts | Item | Concrete deliverable / acceptance check |
|----|------|------------------------------------------|
| 15 | **Query router** | A dedicated router node classifies each query as `structured` / `unstructured` / `out_of_scope` **before** any tool use. Out-of-scope is **declined politely** and NOT answered from general knowledge. Test: "Who is the president of France?" → polite decline. |
| 15 | **Tools w/ Pydantic schemas + descriptions** | ≥4–6 tools, each with a clear name, a description good enough that a human knows when to use it, and a Pydantic input schema. Return values typed too. |
| 10 | **Multi-step reasoning** | Agent chains tools for compound questions (e.g. count REFUND + count COMPLAINT then add). Reasoning visible in CLI. |
| 5  | **CLI w/ reasoning output** | `python main.py` → interactive loop that **prints every tool call + observation**, not just the final answer. |
| 5  | **Max-iterations fallback** | Iteration cap (10–15). If no final answer, return a graceful fallback message (no infinite loop). |

**Test queries to demo (put in README):**
"What categories exist?" · "How many refund requests did we get?" · "Show me 5 examples of the SHIPPING category." · "Summarize how agents respond to complaint intents." · "Show me examples of people wanting their money back." · "What is the distribution of intents in the ACCOUNT category?" · "What's the best CRM software?" (OOS) · "Who is the president of France?" (OOS)

---

## Task 2 — Memory (30 pts)

### 2a. Episodic / conversation memory (20 pts)
- [ ] LangGraph **checkpointer = SqliteSaver** (file-backed, survives restart)
- [ ] `python main.py --session my_session` restores that conversation **after a restart**
- [ ] Follow-ups that reference earlier turns work:
      "Show me 3 examples from REFUND" → "Show me 3 more"; "How many complaints?" → "What about refunds?" → "Total of the last two?"
- Acceptance: run, ask something, quit, rerun with same `--session`, ask "what did I just ask?" → remembers.

### 2b. User profile (10 pts)
- [ ] Persistent **per-user profile** of *distilled facts* (name, frequent topics, preferences) — NOT a message replay
- [ ] Stored **separately** from conversation history (per-user file e.g. `profiles/<user>.json|md`, or a Store)
- [ ] Answers "What do you remember about me?" from the profile
- [ ] Updates naturally as new facts appear; persists across restarts

---

## Task 3 — MCP Server (20 pts)
- [ ] **FastMCP** server exposing **≥3** of the tools as MCP tools
- [ ] README section: how to **start the server** + a **client snippet** that calls one tool
- Acceptance: start server, run the client snippet, get a real result back.

---

## Bonuses (deferred — only if time after 100 core pts)
- **A (+10) Streamlit UI:** chat interface; shows reasoning steps; sidebar session-id input. (uncomment `streamlit` in requirements)
- **B (+10) Query recommender:** on "what should I query next?" → suggest (don't run) using history+profile → refine via chat → execute only on confirm.

---

## Final packing step (do LAST, after implementation + review)
Zip the repo folder excluding `.venv`, `__pycache__`, the real `.env`, caches, and the PDF if not wanted:
```
zip -r "<First Last_First Last>-assignment3.zip" . \
  -x ".venv/*" "*/__pycache__/*" "*.pyc" ".env" ".git/*" ".idea/*" "*.sqlite"
```
Verify the zip contains requirements.txt, README.md, and all source files; that's the submission.
