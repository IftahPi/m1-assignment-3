"""Offline end-to-end evaluation of the full agent loop (router + tools + answer).

NOT a unit test: it makes live Nebius API calls and is non-deterministic. Run it
manually after changing the graph, prompts, or tools:

    .venv/bin/python agent_eval.py

Each case asserts properties of the agent's *final answer*: that it did not give
up (the fallback message), that out-of-scope questions are declined, and that
concrete answers contain the expected facts (counts, names) the tools provide.
"""

from typing import NamedTuple

from langchain_core.messages import HumanMessage

from agent.graph import DECLINE_MESSAGE, FALLBACK_MESSAGE, build_graph

_CONFIG: dict = {"recursion_limit": 50}


class AgentEvalCase(NamedTuple):
    """One end-to-end case: a query plus expectations on the final answer."""

    query: str
    must_contain: tuple[str, ...] = ()
    expect_decline: bool = False


CASES: list[AgentEvalCase] = [
    AgentEvalCase("How many refund requests did we get?", must_contain=("2992",)),
    AgentEvalCase(
        "What categories exist in the dataset?",
        must_contain=("ACCOUNT", "REFUND", "SHIPPING"),
    ),
    AgentEvalCase(
        "What is the distribution of intents in the ACCOUNT category?",
        must_contain=("edit_account",),
    ),
    AgentEvalCase(
        "How many complaints did we get, and how many refunds? What is the total of the two?",
        must_contain=("3992",),
    ),
    # These two previously looped to the fallback; the dedupe guard must let them finish.
    AgentEvalCase("Show me 3 examples from the SHIPPING category."),
    AgentEvalCase("Show me examples of people wanting their money back."),
    # Assignment phrasing: "SHIPPING intent" -- but SHIPPING is a CATEGORY. The agent must
    # disambiguate (uppercase word -> category) and still return useful examples.
    AgentEvalCase("Show me 3 examples from the SHIPPING intent.", must_contain=("SHIPPING",)),
    AgentEvalCase("Who is the president of France?", expect_decline=True),
    AgentEvalCase("What's the best CRM software for handling complaints?", expect_decline=True),
]


def _final_answer(graph: object, query: str) -> str:
    """Run the graph for one query and return the text of the final message."""
    state = {"messages": [HumanMessage(content=query)], "route": "", "iterations": 0}
    result = graph.invoke(state, config=_CONFIG)
    return str(result["messages"][-1].content)


def _check(case: AgentEvalCase, answer: str) -> tuple[bool, str]:
    """Return (passed, reason) for one case's final answer."""
    if case.expect_decline:
        if answer == DECLINE_MESSAGE:
            return True, "declined as expected"
        return False, "expected a polite decline"
    if answer == FALLBACK_MESSAGE:
        return False, "gave up (hit the fallback)"
    missing = [s for s in case.must_contain if s.lower() not in answer.lower()]
    if missing:
        return False, f"missing expected text: {missing}"
    return True, "answered with the expected content"


def evaluate_agent(cases: list[AgentEvalCase] = CASES) -> int:
    """Run every case through the live agent, print results, and return the number passed."""
    graph = build_graph()
    passed = 0
    for case in cases:
        answer = _final_answer(graph, case.query)
        ok, reason = _check(case, answer)
        if ok:
            passed += 1
        mark = "✅" if ok else "❌"
        print(f"{mark} {reason:32s} | {case.query}")
        if not ok:
            print(f"     got: {answer[:140]!r}")
    print(f"\nAgent eval: {passed}/{len(cases)} ({100 * passed / len(cases):.0f}%)")
    return passed


if __name__ == "__main__":
    evaluate_agent()
