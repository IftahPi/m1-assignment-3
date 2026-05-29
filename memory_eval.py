"""Live memory acceptance harness — for LLM-as-judge evaluation.

Runs a scripted sequence of (user_id, session_id, inputs) scenarios against
the real agent: SqliteSaver + per-user Markdown profile + the four-label
router. Prints the full trace of each scenario, AND prints each user's
profile file content after every scenario, so a reviewer (human or LLM)
can read the trace, compare against the rubric below, and judge.

NOT a unit test. Makes live Nebius API calls (~3-5 minutes).

State is isolated from your normal CLI runs:
- checkpoints DB: ``checkpoints_memory_eval.sqlite`` (deleted at start)
- profiles dir: ``profiles_memory_eval/`` (deleted at start)
Both are gitignored.

Run:
    python memory_eval.py
"""

import shutil
from pathlib import Path
from typing import Callable, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import profile as profile_module
from agent.graph import build_graph
from agent.profile import get_personal_info, save_profile, summarize_session
from nebius_client import GENERATOR_MODEL, make_llm

_EVAL_DIR: Path = Path(__file__).resolve().parent
_CHECKPOINT_DB: Path = _EVAL_DIR / "checkpoints_memory_eval.sqlite"
_PROFILES_OVERRIDE: Path = _EVAL_DIR / "profiles_memory_eval"


RUBRIC: str = """\
MEMORY EVALUATION RUBRIC — for an LLM-as-judge reading the scenarios below.

For each scenario, mark every criterion ✓ pass / ✗ fail / ⚠ partial.

HARD criteria — any ✗ here = the scenario FAILS:
  H1. ROUTE correctness: the router's label matches the scenario's intent.
        - Self-introductions ("my name is X") → structured (a statement, not
          a personal question; the agent should acknowledge).
        - Recall ("what do you remember about me?", "what's my name?") →
          personal.
  H2. PERSONAL-ROUTE GROUNDING: when the route is `personal`, the agent
      calls the `get_personal_info` tool, and its final answer is grounded
      ONLY in the tool's returned content. No facts invented from world
      knowledge; no facts from the dataset tools.
  H3. USER ISOLATION: when reading Alice's profile, no information from
      Bob's profile appears (or vice versa). The closure-scoped tool must
      refuse cross-user requests if any were attempted.
  H4. EMPTY PROFILE BEHAVIOR: for a brand-new user with no profile yet, the
      personal route answers with "I don't have that on file" (or similar)
      and does NOT invent a name.
  H5. SUMMARY DISTILLATION: the profile file written at session end captures
      the durable facts shared in that session (name, interests, prefs) and
      does NOT contain a transcript or a recap of one-off questions.

SOFT criteria — quality signals, not auto-failing:
  S1. The model honors stated preferences (e.g. "prefers concise answers").
  S2. The tool is called once, with the correct user_id.
  S3. The summary keeps prior facts unless contradicted; contradictions are
      replaced, not duplicated (S6 below explicitly tests this).
"""


class Scenario(NamedTuple):
    """One memory-eval scenario."""

    label: str
    user_id: str
    session_id: str
    inputs: list[str]
    why: str


SCENARIOS: list[Scenario] = [
    Scenario(
        label="S1: Alice — intro (name + interest + preference) + data Q",
        user_id="alice",
        session_id="alice_s1",
        inputs=[
            "Hi, my name is Alice. I'm interested in REFUND data and I prefer concise answers.",
            "How many refund requests did we get?",
        ],
        why=(
            "Plants three durable facts (name / interest / preference) AND a real data "
            "question. The intro should route structured (statement, not personal); the "
            "data question should route structured. At session-end summary should write a "
            "profile capturing all three facts."
        ),
    ),
    Scenario(
        label="S2: Alice — NEW session, recall via personal route",
        user_id="alice",
        session_id="alice_s2",
        inputs=["What do you remember about me?"],
        why=(
            "The classic cross-session recall test. Different session_id, same user_id. "
            "Router should classify personal; personal node must call get_personal_info; "
            "answer must mention Alice's name, interest, or preference — and be terse "
            "because the profile says she prefers concise answers."
        ),
    ),
    Scenario(
        label="S3: Bob — fresh user, only shares his name",
        user_id="bob",
        session_id="bob_s1",
        inputs=["My name is Bob."],
        why=(
            "Verifies a SECOND user can have an independent profile. Self-introduction "
            "should route structured; summary should write profiles/bob.md with at least "
            "the name."
        ),
    ),
    Scenario(
        label="S4: Bob — new session, asks for his name",
        user_id="bob",
        session_id="bob_s2",
        inputs=["What's my name?"],
        why=(
            "Bob's cross-session recall. Personal route must read Bob's profile (NOT "
            "Alice's). Answer must be 'Bob', proving user-scoped isolation works."
        ),
    ),
    Scenario(
        label="S5: Carl — brand-new user, no profile yet",
        user_id="carl",
        session_id="carl_s1",
        inputs=["What do you remember about me?"],
        why=(
            "Empty-profile behavior. Personal route fires; tool returns empty profile; "
            "the agent must say 'I don't have that on file' (or equivalent) and MUST NOT "
            "invent a name."
        ),
    ),
    Scenario(
        label="S6: Alice — verify Bob's profile didn't leak; test contradiction handling",
        user_id="alice",
        session_id="alice_s3",
        inputs=[
            "What's my name?",
            "Actually, my name is changed — call me Alicia from now on.",
        ],
        why=(
            "Two checks. (a) The first question must still recall 'Alice' (not 'Bob' — "
            "user isolation across the whole eval). (b) The statement plants a "
            "contradicting fact (Alicia vs Alice). After session-end summary, the profile "
            "should REPLACE 'Alice' with 'Alicia', not keep both — testing the explicit "
            "contradiction rule in SUMMARY_SYSTEM_PROMPT."
        ),
    ),
]


def _scripted_input(lines: list[str]) -> Callable[[str], str]:
    """A fake ``input`` that yields the given lines then 'quit'."""
    iterator = iter([*lines, "quit"])
    return lambda _prompt: next(iterator)


def _render_message(message: object, output_fn: Callable[[str], None]) -> None:
    """Print a single message in FULL for the judge."""
    if isinstance(message, AIMessage) and message.tool_calls:
        thought = str(message.content or "").strip()
        if thought:
            output_fn(f"  💭 THOUGHT: {thought}")
        for call in message.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items() if v is not None)
            output_fn(f"  🔧 CALL:    {call['name']}({args})")
    elif isinstance(message, ToolMessage):
        output_fn("  📊 RESULT:")
        for line in str(message.content).splitlines() or [str(message.content)]:
            output_fn(f"     {line}")
    elif isinstance(message, AIMessage):
        output_fn("")
        output_fn("  🤖 FINAL ANSWER:")
        for line in str(message.content).splitlines() or [str(message.content)]:
            output_fn(f"     {line}")
        output_fn("")


def _trace_node_updates(node_name: str, update: object, output_fn: Callable[[str], None]) -> None:
    if not isinstance(update, dict):
        return
    if node_name == "router" and update.get("route"):
        output_fn(f"  🧭 ROUTE:   {update['route']}")
        return
    for message in update.get("messages", []):
        _render_message(message, output_fn)


def _run_scenario(scenario: Scenario, output_fn: Callable[[str], None]) -> None:
    """Run one scenario end-to-end including end-of-session summary."""
    output_fn("=" * 92)
    output_fn(f"{scenario.label}")
    output_fn(f"  user_id   = {scenario.user_id!r}")
    output_fn(f"  session_id= {scenario.session_id!r}")
    output_fn(f"  why       : {scenario.why}")
    output_fn("-" * 92)

    config = {"configurable": {"thread_id": scenario.session_id}, "recursion_limit": 50}

    with SqliteSaver.from_conn_string(str(_CHECKPOINT_DB)) as saver:
        graph = build_graph(checkpointer=saver)
        for user_line in scenario.inputs:
            output_fn(f"\nYou> {user_line}")
            initial = {
                "messages": [HumanMessage(content=user_line)],
                "route": "",
                "iterations": 0,
                "user_id": scenario.user_id,
            }
            for chunk in graph.stream(initial, config=config, stream_mode="updates"):
                for node_name, update in chunk.items():
                    _trace_node_updates(node_name, update, output_fn)

        # End-of-session summary distillation (mirrors main.py's _distill_profile_on_exit).
        state = graph.get_state({"configurable": {"thread_id": scenario.session_id}})
        messages = state.values.get("messages", [])
        if any(isinstance(m, HumanMessage) for m in messages):
            prior = get_personal_info(scenario.user_id)
            summary_llm = make_llm(GENERATOR_MODEL, temperature=0.2)
            updated = summarize_session(messages, prior, summary_llm)
            save_profile(scenario.user_id, updated)
            output_fn(f"\n💾 Updated profile for user '{scenario.user_id}'")

    profile_text = get_personal_info(scenario.user_id) or "(empty)"
    output_fn(f"\n📝 PROFILE STATE for user {scenario.user_id!r} after this scenario:")
    for line in profile_text.splitlines() or ["(empty)"]:
        output_fn(f"   {line}")
    output_fn("")


def main() -> None:
    """Reset isolated state, run every scenario, dump final per-user profiles."""
    # Isolate the eval's state.
    _CHECKPOINT_DB.unlink(missing_ok=True)
    if _PROFILES_OVERRIDE.exists():
        shutil.rmtree(_PROFILES_OVERRIDE)
    profile_module._PROFILES_DIR = _PROFILES_OVERRIDE

    print(f"# Memory eval — {len(SCENARIOS)} scenarios")
    print(f"# checkpoint db: {_CHECKPOINT_DB.name}")
    print(f"# profiles dir : {_PROFILES_OVERRIDE.name}/")
    print()
    print(RUBRIC)
    print()

    for scenario in SCENARIOS:
        _run_scenario(scenario, print)

    print("=" * 92)
    print("FINAL PROFILE STATE for every user:")
    for user_id in sorted({s.user_id for s in SCENARIOS}):
        print(f"\n--- profiles_memory_eval/{user_id}.md ---")
        print(get_personal_info(user_id) or "(empty)")


if __name__ == "__main__":
    main()
