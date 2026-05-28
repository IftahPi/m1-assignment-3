"""Offline evaluation of the query router against the assignment's example questions.

This is NOT a unit test: it makes live Nebius API calls and is non-deterministic.
Run it manually whenever the router prompt changes:

    .venv/bin/python router_eval.py

Each case pairs an example question with the label the router *should* choose.
"""

from typing import NamedTuple

from agent.router import classify_query


class EvalCase(NamedTuple):
    """A single router evaluation case: a question and its expected label."""

    query: str
    expected: str


CASES: list[EvalCase] = [
    # structured — concrete, data-driven answers
    EvalCase("What categories exist in the dataset?", "structured"),
    EvalCase("How many refund requests did we get?", "structured"),
    EvalCase("Show me 3 examples from the SHIPPING intent.", "structured"),
    EvalCase("Show me 5 examples of the SHIPPING category.", "structured"),
    EvalCase("What is the distribution of intents in the ACCOUNT category?", "structured"),
    EvalCase("Show me examples of people wanting their money back.", "structured"),
    # unstructured — open-ended summarization
    EvalCase("Summarize the FEEDBACK category.", "unstructured"),
    EvalCase(
        "How do customer service representatives typically respond to cancellation requests?",
        "unstructured",
    ),
    EvalCase("Summarize how agents respond to complaint intents.", "unstructured"),
    # out_of_scope — unrelated to the dataset, must be declined
    EvalCase("Who won the 2024 Champions League?", "out_of_scope"),
    EvalCase("Write me a poem about customer service.", "out_of_scope"),
    EvalCase("What's the best CRM software for handling complaints?", "out_of_scope"),
    EvalCase("Who is the president of France?", "out_of_scope"),
]


def evaluate_router(cases: list[EvalCase] = CASES) -> int:
    """Classify each case, print a per-case result line, and return the number correct."""
    correct = 0
    for case in cases:
        decision = classify_query(case.query)
        is_hit = decision.route == case.expected
        if is_hit:
            correct += 1
        mark = "✅" if is_hit else "❌"
        print(f"{mark} expected={case.expected:13s} got={decision.route:13s} | {case.query}")
    print(f"\nAccuracy: {correct}/{len(cases)} ({100 * correct / len(cases):.0f}%)")
    return correct


if __name__ == "__main__":
    evaluate_router()
