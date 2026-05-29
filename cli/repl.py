"""Interactive REPL that runs the agent graph and shows its reasoning steps.

For each turn it streams the graph and prints the router decision, every tool
call and its observation, and finally the agent's answer — not just the answer.
The graph and the input/output functions are injected so the loop is testable
without an LLM.
"""

from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph

WELCOME_MESSAGE: str = """\
👋  Hi! I'm a data-analyst assistant for the Bitext Customer Service dataset —
    26,872 real customer support messages and agent replies across 11 categories
    (ACCOUNT, ORDER, REFUND, SHIPPING, FEEDBACK, and more).

    Ask me about the data, for example:
      • How many refund requests did we get?
      • Show me 5 examples from the SHIPPING category.
      • What is the distribution of intents in the ACCOUNT category?
      • Summarize how agents respond to complaints.

    As I work, my reasoning streams below — here is what each symbol means:
      🧭  router decision — how I classified your question
      💭  thought         — what I'm about to do, and why
      🔧  tool call       — the function I'm calling and its arguments
      📊  observation     — what the tool returned
      🤖  final answer    — my reply to you

    I only answer questions about this dataset; anything else I'll set aside.
    Type 'quit' or 'exit' to leave.
"""

_RECURSION_LIMIT: int = 50


def _stream_config(session_id: str) -> dict:
    """Build the per-turn graph config: session keys the checkpoint thread."""
    return {"configurable": {"thread_id": session_id}, "recursion_limit": _RECURSION_LIMIT}


def _format_args(args: dict) -> str:
    """Render tool-call arguments, dropping the ones left as None."""
    return ", ".join(f"{key}={value!r}" for key, value in args.items() if value is not None)


def _truncate(text: str, limit: int = 300) -> str:
    """Shorten long observations for readable console output."""
    return text if len(text) <= limit else text[:limit] + "…"


def _render_message(message: object, output_fn: Callable[[str], None]) -> None:
    """Print a single graph message as a reasoning step or the final answer."""
    if isinstance(message, AIMessage) and message.tool_calls:
        # When the model emits text alongside tool_calls, it's the pre-action "thought".
        thought = str(message.content or "").strip()
        if thought:
            output_fn(f"  💭 {thought}")
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


def _run_turn(
    graph: object, query: str, session_id: str, output_fn: Callable[[str], None]
) -> None:
    """Stream one agent turn and render each step."""
    state = {"messages": [HumanMessage(content=query)], "route": "", "iterations": 0}
    for chunk in graph.stream(state, config=_stream_config(session_id), stream_mode="updates"):
        for node_name, update in chunk.items():
            _render_update(node_name, update, output_fn)


def run_repl(
    graph: object | None = None,
    session_id: str = "default",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive REPL loop over the agent graph.

    Args:
        graph: A compiled agent graph. Defaults to ``build_graph()``.
        session_id: Keys the checkpointed conversation thread; the same id
            on a later run resumes the same conversation.
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

        _run_turn(graph, query, session_id, output_fn)
