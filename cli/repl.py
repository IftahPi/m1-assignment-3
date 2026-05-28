"""Interactive REPL loop for the query router."""

from typing import Callable

from agent.router import classify_query
from agent.schemas import RouteDecision

WELCOME_MESSAGE: str = (
    "👋  Hi! I'm a data-analyst assistant for the Bitext Customer Service dataset —\n"
    "    26,872 real customer support messages and agent replies across 11 categories\n"
    "    (ACCOUNT, ORDER, REFUND, SHIPPING, FEEDBACK, and more).\n\n"
    "    Ask me about the data, for example:\n"
    "      • How many refund requests did we get?\n"
    "      • Show me 5 examples from the SHIPPING category.\n"
    "      • What is the distribution of intents in the ACCOUNT category?\n"
    "      • Summarize how agents respond to complaints.\n\n"
    "    I only answer questions about this dataset; anything else I'll set aside.\n"
    "    Type 'quit' or 'exit' to leave.\n"
)


def run_repl(
    classify: Callable[[str], RouteDecision] = classify_query,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive REPL loop.

    Args:
        classify: Function to classify a query into a route. Defaults to classify_query.
        input_fn: Function to read user input. Defaults to built-in input.
        output_fn: Function to print output. Defaults to built-in print.
    """
    output_fn(WELCOME_MESSAGE)
    while True:
        try:
            query = input_fn("You> ")
        except (EOFError, KeyboardInterrupt):
            break

        if query in ("quit", "exit"):
            break
        if not query.strip():
            continue  # empty input: re-prompt rather than exit

        decision = classify(query)
        output_fn(f"[router] {decision.route} — {decision.reason}")
