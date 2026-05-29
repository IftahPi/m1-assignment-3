"""Tests for the interactive REPL, using a fake graph (no LLM, no network)."""

from langchain_core.messages import AIMessage, ToolMessage

from cli.repl import WELCOME_MESSAGE, run_repl


class FakeGraph:
    """A stand-in graph whose .stream() yields canned update chunks."""

    def __init__(self, chunks: list | None = None) -> None:
        self.chunks = chunks or []
        self.stream_calls = 0
        self.last_config: dict | None = None

    def stream(self, state, config=None, stream_mode=None):
        self.stream_calls += 1
        self.last_config = config
        return iter(self.chunks)


def _scripted_input(lines: list[str]):
    it = iter(lines)
    return lambda prompt: next(it)


def test_repl_prints_welcome_on_start():
    graph = FakeGraph()
    outputs: list[str] = []
    run_repl(graph=graph, input_fn=_scripted_input(["quit"]), output_fn=outputs.append)

    assert outputs[0] == WELCOME_MESSAGE
    assert "dataset" in WELCOME_MESSAGE.lower()


def test_repl_passes_session_id_as_thread_id_in_graph_config():
    """The session_id arg flows through to the graph as configurable.thread_id."""
    chunks = [{"agent": {"messages": [AIMessage(content="ok")]}}]
    graph = FakeGraph(chunks)
    run_repl(
        graph=graph,
        session_id="my_session_123",
        input_fn=_scripted_input(["hi", "quit"]),
        output_fn=lambda _t: None,
    )
    assert graph.last_config is not None
    assert graph.last_config["configurable"]["thread_id"] == "my_session_123"


def test_repl_renders_router_tool_observation_and_answer():
    chunks = [
        {"router": {"route": "structured"}},
        {"agent": {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": "count_records", "args": {"category": "REFUND"},
                         "id": "c1", "type": "tool_call"}],
        )]}},
        {"tools": {"messages": [ToolMessage(content="2992", tool_call_id="c1")]}},
        {"agent": {"messages": [AIMessage(content="We received 2992 refund requests.")]}},
    ]
    graph = FakeGraph(chunks)
    outputs: list[str] = []
    run_repl(graph=graph, input_fn=_scripted_input(["How many refunds?", "quit"]),
             output_fn=outputs.append)

    text = "\n".join(outputs)
    assert "router → structured" in text          # router decision shown
    assert "count_records" in text                 # tool call shown
    assert "REFUND" in text                         # tool args shown
    assert "2992" in text                           # observation shown
    assert "We received 2992 refund requests." in text  # final answer shown


def test_welcome_message_includes_trace_symbol_legend():
    """The welcome banner explains every symbol the reasoning trace uses."""
    legend = WELCOME_MESSAGE
    for symbol in ("🧭", "💭", "🔧", "📊", "🤖"):
        assert symbol in legend, f"welcome banner is missing symbol: {symbol}"
    for label in ("router", "thought", "tool", "observation", "answer"):
        assert label in legend.lower(), f"welcome banner is missing label: {label!r}"


def test_repl_renders_thought_before_tool_call_when_content_present():
    """A non-empty content alongside tool_calls is rendered as a thought BEFORE the tool call."""
    chunks = [
        {"agent": {"messages": [AIMessage(
            content="I need to count refund requests in the dataset.",
            tool_calls=[{"name": "count_records", "args": {"category": "REFUND"},
                         "id": "c1", "type": "tool_call"}],
        )]}},
    ]
    graph = FakeGraph(chunks)
    outputs: list[str] = []
    run_repl(graph=graph, input_fn=_scripted_input(["go", "quit"]),
             output_fn=outputs.append)

    text = "\n".join(outputs)
    assert "💭" in text
    assert "I need to count refund requests in the dataset." in text
    # The thought is rendered BEFORE the tool call line.
    assert text.index("I need to count refund") < text.index("count_records")


def test_repl_skips_thought_when_content_is_empty():
    """Empty content alongside tool_calls must NOT print a stray thought line."""
    chunks = [
        {"agent": {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": "count_records", "args": {"category": "REFUND"},
                         "id": "c1", "type": "tool_call"}],
        )]}},
    ]
    graph = FakeGraph(chunks)
    outputs: list[str] = []
    run_repl(graph=graph, input_fn=_scripted_input(["go", "quit"]),
             output_fn=outputs.append)

    # outputs[0] is the welcome banner, which legitimately contains 💭 in its
    # symbol legend. Check that no PER-STEP render line emitted a 💭 thought.
    assert all("💭" not in line for line in outputs[1:])


def test_repl_skips_empty_and_whitespace_input():
    graph = FakeGraph([{"agent": {"messages": [AIMessage(content="hi")]}}])
    outputs: list[str] = []
    run_repl(graph=graph, input_fn=_scripted_input(["", "   ", "quit"]),
             output_fn=outputs.append)

    assert graph.stream_calls == 0  # never ran the graph for blank input


def test_repl_exits_on_quit_without_running_graph():
    graph = FakeGraph()
    run_repl(graph=graph, input_fn=_scripted_input(["quit"]), output_fn=lambda _t: None)
    assert graph.stream_calls == 0


def test_repl_handles_eof():
    graph = FakeGraph()

    def raise_eof(prompt: str) -> str:
        raise EOFError()

    run_repl(graph=graph, input_fn=raise_eof, output_fn=lambda _t: None)
    assert graph.stream_calls == 0


def test_repl_handles_keyboard_interrupt():
    graph = FakeGraph()

    def raise_interrupt(prompt: str) -> str:
        raise KeyboardInterrupt()

    run_repl(graph=graph, input_fn=raise_interrupt, output_fn=lambda _t: None)
    assert graph.stream_calls == 0
