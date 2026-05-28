"""Test the interactive REPL loop."""

from agent.schemas import RouteDecision
from cli.repl import WELCOME_MESSAGE, run_repl


def test_repl_prints_welcome_on_start():
    """The REPL greets the user with the welcome message before reading any input."""
    def fake_classify(query: str) -> RouteDecision:
        return RouteDecision(route="structured", reason="Test.")

    def fake_input(prompt: str) -> str:
        return "quit"

    output_lines: list[str] = []
    run_repl(classify=fake_classify, input_fn=fake_input, output_fn=output_lines.append)

    assert output_lines[0] == WELCOME_MESSAGE
    assert "dataset" in WELCOME_MESSAGE.lower()


def test_repl_basic_flow():
    """Test that the REPL reads a query, classifies it, prints the result, and handles quit."""
    # Fixed classify function
    def fake_classify(query: str) -> RouteDecision:
        if query == "How many orders?":
            return RouteDecision(route="structured", reason="Asking for a count.")
        return RouteDecision(route="out_of_scope", reason="Unknown.")

    # Scripted input
    inputs = ["How many orders?", "quit"]
    input_iter = iter(inputs)

    def fake_input(prompt: str) -> str:
        return next(input_iter)

    # Capture output
    output_lines = []

    def fake_output(text: str) -> None:
        output_lines.append(text)

    run_repl(classify=fake_classify, input_fn=fake_input, output_fn=fake_output)

    # Verify output contains the route and reason
    output_text = "\n".join(output_lines)
    assert "structured" in output_text
    assert "Asking for a count." in output_text


def test_repl_handles_eof():
    """Test that the REPL exits gracefully on EOFError."""
    def fake_classify(query: str) -> RouteDecision:
        return RouteDecision(route="structured", reason="Test.")

    def fake_input_eof(prompt: str) -> str:
        raise EOFError()

    output_lines = []

    def fake_output(text: str) -> None:
        output_lines.append(text)

    # Should not raise; should exit cleanly
    run_repl(classify=fake_classify, input_fn=fake_input_eof, output_fn=fake_output)


def test_repl_handles_keyboard_interrupt():
    """Test that the REPL exits gracefully on KeyboardInterrupt."""
    def fake_classify(query: str) -> RouteDecision:
        return RouteDecision(route="structured", reason="Test.")

    def fake_input_interrupt(prompt: str) -> str:
        raise KeyboardInterrupt()

    output_lines = []

    def fake_output(text: str) -> None:
        output_lines.append(text)

    # Should not raise; should exit cleanly
    run_repl(classify=fake_classify, input_fn=fake_input_interrupt, output_fn=fake_output)


def test_repl_exit_command():
    """Test that 'exit' terminates the REPL."""
    def fake_classify(query: str) -> RouteDecision:
        return RouteDecision(route="structured", reason="Test.")

    inputs = ["exit"]
    input_iter = iter(inputs)

    def fake_input(prompt: str) -> str:
        return next(input_iter)

    output_lines = []

    def fake_output(text: str) -> None:
        output_lines.append(text)

    run_repl(classify=fake_classify, input_fn=fake_input, output_fn=fake_output)

    # Should have exited without error


def test_repl_empty_input_is_skipped():
    """Empty/whitespace input re-prompts (is not classified); loop continues until quit."""
    classified_queries = []

    def fake_classify(query: str) -> RouteDecision:
        classified_queries.append(query)
        return RouteDecision(route="structured", reason="Test.")

    inputs = ["", "   ", "quit"]
    input_iter = iter(inputs)

    def fake_input(prompt: str) -> str:
        return next(input_iter)

    output_lines = []

    def fake_output(text: str) -> None:
        output_lines.append(text)

    run_repl(classify=fake_classify, input_fn=fake_input, output_fn=fake_output)

    # Empty and whitespace-only inputs are skipped, never classified.
    assert classified_queries == []
    # Only the welcome banner is printed; no router output lines.
    assert not any(line.startswith("[router]") for line in output_lines)
