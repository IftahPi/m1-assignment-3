"""Interactive REPL that runs the agent graph and shows its reasoning steps.

For each turn it streams the graph and prints the router decision, every tool
call and its observation, and finally the agent's answer — not just the answer.
The graph and the input/output functions are injected so the loop is testable
without an LLM.
"""

from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph

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

_STREAM_CONFIG: dict = {"recursion_limit": 50}


def _format_args(args: dict) -> str:
    """Render tool-call arguments, dropping the ones left as None."""
    return ", ".join(f"{key}={value!r}" for key, value in args.items() if value is not None)


def _truncate(text: str, limit: int = 300) -> str:
    """Shorten long observations for readable console output."""
    return text if len(text) <= limit else text[:limit] + "…"


def _render_message(message: object, output_fn: Callable[[str], None]) -> None:
    """Print a single graph message as a reasoning step or the final answer."""
    if isinstance(message, AIMessage) and message.tool_calls:
        for call in message.tool_calls:
            output_fn(f"  🔧 {call['name']}({_format_args(call['args'])})")
    elif isinstance(message, ToolMessage):
        output_fn(f"  📊 {_truncate(str(message.content))}")
    elif isinstance(message, AIMessage):
        output_fn(f"\n🤖 {message.content}\n")


def _render_update(node_name: str, update: object, output_fn: Callable[[str], None]) -> None:
    """Render one streamed node update (router label or messages)."""
    if not isinstance(update, dict):
        return
    if node_name == "router" and update.get("route"):
        output_fn(f"  🧭 router → {update['route']}")
        return
    for message in update.get("messages", []):
        _render_message(message, output_fn)


def _run_turn(graph: object, query: str, output_fn: Callable[[str], None]) -> None:
    """Stream one agent turn and render each step."""
    state = {"messages": [HumanMessage(content=query)], "route": "", "iterations": 0}
    for chunk in graph.stream(state, config=_STREAM_CONFIG, stream_mode="updates"):
        for node_name, update in chunk.items():
            _render_update(node_name, update, output_fn)


def run_repl(
    graph: object | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive REPL loop over the agent graph.

    Args:
        graph: A compiled agent graph. Defaults to ``build_graph()``.
        input_fn: Reads a line of user input. Defaults to ``input``.
        output_fn: Prints a line of output. Defaults to ``print``.
    """
    if graph is None:
        graph = build_graph()
    output_fn(WELCOME_MESSAGE)
    while True:
        try:
            query = input_fn("You> ")
        except (EOFError, KeyboardInterrupt):
            break

        if query in ("quit", "exit"):
            break
        if not query.strip():
            continue

        _run_turn(graph, query, output_fn)
