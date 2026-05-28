"""End-to-end agent trace dump for LLM-as-judge evaluation.

NOT a pass/fail script and NOT a unit test. It runs each query through the live
graph and prints the FULL trace — router decision, every tool call and its
observation in full, and the final answer in full — to standard output.

Why no automatic grading: agent-answer quality is open-ended. Substring checks
pass false-positive answers like "I am looking up examples for you..." that are
syntactically about the topic but contain none of the data. The honest
evaluation is a reviewer (human or an LLM-as-judge) reading the trace and
deciding whether the final answer actually contains the data, addresses the
question, and stays in scope (per the Week 4 lecture: "don't grade the path,
grade the outcome — and model-based judgement is the right tool when the
outcome can't be code-checked").

Usage:
    python agent_eval.py                # run every case
    python agent_eval.py shipping       # only cases whose query contains 'shipping'
    python agent_eval.py "money back"   # quote multi-word filters
"""

import sys
from typing import Callable, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph
from nebius_client import GENERATOR_MODEL, ROUTER_MODEL

_CONFIG: dict = {"recursion_limit": 50}

RUBRIC: str = """\
EVALUATION RUBRIC — for an LLM-as-judge reading the traces below.

For each case, mark every criterion ✓ pass / ✗ fail / ⚠ partial.

HARD criteria — ANY ✗ here = the case FAILS overall:
  H1. A FINAL ANSWER was produced (not the fallback "I couldn't work that out…"
      message and not silence).
  H2. ROUTE correctness:
        structured / unstructured  → routed there AND answered;
        out_of_scope               → routed there AND politely declined.
  H3. IN-SCOPE answers contain the ACTUAL DATA from the tool result —
      the number, the distribution, the example messages — NOT merely a
      sentence narrating intent ("I am looking up…", "I will search for…").
  H4. OUT-OF-SCOPE answers contain the polite decline and DO NOT use the
      LLM's world knowledge to answer the question.

SOFT criteria — quality signals; logged but do not auto-fail:
  S1. Tool choice & arguments are reasonable for the question.
  S2. Multi-step questions (e.g. "X and Y, what's the total?") return BOTH
      component results AND the combined answer, computed by the LLM.
  S3. Tool calls are economical — no redundant identical calls; the
      force_answer guard should not have to fire on routine queries.
  S4. Answer is concise and grounded in the tool result; not padded with
      filler or fabricated detail.
  S5. The model states the assumption it made when the user's wording was
      ambiguous (e.g. "assumed category 'REFUND'").

Verdict per case: one paragraph — PASS or FAIL, which criteria, ≤2-line
justification quoting the trace where useful. End with a summary noting
any regressions vs. previous runs.
"""


class TraceCase(NamedTuple):
    """One query to trace; the expected route is informational, not asserted."""

    query: str
    expected_route: str


CASES: list[TraceCase] = [
    # structured — concrete, data-driven
    TraceCase("What categories exist in the dataset?", "structured"),
    TraceCase("How many refund requests did we get?", "structured"),
    TraceCase("Show me 3 examples from the SHIPPING category.", "structured"),
    TraceCase("Show me 3 examples from the SHIPPING intent.", "structured"),
    TraceCase("What is the distribution of intents in the ACCOUNT category?", "structured"),
    TraceCase("Show me examples of people wanting their money back.", "structured"),
    TraceCase(
        "How many complaints did we get, and how many refunds? What is the total of the two?",
        "structured",
    ),
    # unstructured — open-ended summarization
    TraceCase("Summarize how agents respond to complaints.", "unstructured"),
    TraceCase("Summarize the FEEDBACK category.", "unstructured"),
    # out_of_scope — must be declined, not answered
    TraceCase("Who is the president of France?", "out_of_scope"),
    TraceCase("What's the best CRM software for handling complaints?", "out_of_scope"),
]


def _format_args(args: dict) -> str:
    """Render tool-call arguments in a readable, full form."""
    return ", ".join(f"{key}={value!r}" for key, value in args.items() if value is not None)


def _render_message(message: object, output_fn: Callable[[str], None]) -> None:
    """Print a single message in FULL (no truncation) for the judge's benefit."""
    if isinstance(message, AIMessage) and message.tool_calls:
        thought = str(message.content or "").strip()
        if thought:
            output_fn(f"  💭 THOUGHT: {thought}")
        for call in message.tool_calls:
            output_fn(f"  🔧 CALL:    {call['name']}({_format_args(call['args'])})")
    elif isinstance(message, ToolMessage):
        # Print the full observation; tool results are what the judge needs to see.
        output_fn("  📊 RESULT:")
        for line in str(message.content).splitlines() or [str(message.content)]:
            output_fn(f"     {line}")
    elif isinstance(message, AIMessage):
        output_fn("")
        output_fn("  🤖 FINAL ANSWER:")
        for line in str(message.content).splitlines() or [str(message.content)]:
            output_fn(f"     {line}")
        output_fn("")


def _trace_case(graph: object, case: TraceCase, output_fn: Callable[[str], None]) -> None:
    """Run one case and print every step in full."""
    output_fn("=" * 88)
    output_fn(f"QUERY (expected route: {case.expected_route}):")
    output_fn(f"  {case.query}")
    output_fn("-" * 88)

    state = {"messages": [HumanMessage(content=case.query)], "route": "", "iterations": 0}
    for chunk in graph.stream(state, config=_CONFIG, stream_mode="updates"):
        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue
            if node_name == "router" and update.get("route"):
                output_fn(f"  🧭 ROUTE:   {update['route']}")
                continue
            for message in update.get("messages", []):
                _render_message(message, output_fn)
    output_fn("")


def print_traces(
    cases: list[TraceCase] = CASES,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run every case through the live agent and print the full trace of each."""
    output_fn(f"# Agent trace dump")
    output_fn(f"# router    = {ROUTER_MODEL}")
    output_fn(f"# generator = {GENERATOR_MODEL}")
    output_fn(f"# {len(cases)} case(s)")
    output_fn("")
    output_fn(RUBRIC)
    output_fn("")
    graph = build_graph()
    for case in cases:
        _trace_case(graph, case, output_fn)


def _filter_cases(substring: str) -> list[TraceCase]:
    needle = substring.lower()
    return [case for case in CASES if needle in case.query.lower()]


if __name__ == "__main__":
    selected = CASES if len(sys.argv) < 2 else _filter_cases(sys.argv[1])
    if not selected:
        print(f"No cases match filter: {sys.argv[1]!r}", file=sys.stderr)
        sys.exit(1)
    print_traces(selected)
