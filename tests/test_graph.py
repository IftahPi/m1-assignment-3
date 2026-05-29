"""Tests for the agent graph, with the LLM boundary mocked.

The router (classify_query) and the generator (make_llm) are patched so the graph
runs deterministically and offline. The tools run for real against the dataset.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import graph as graph_module
from agent.graph import DECLINE_MESSAGE, FALLBACK_MESSAGE, build_graph

CONFIG = {"recursion_limit": 50}


def _initial_state(text: str) -> dict:
    return {"messages": [HumanMessage(content=text)], "route": "", "iterations": 0}


def _tool_call_message(msg_id: str, category: str = "REFUND") -> AIMessage:
    """An AIMessage asking to run count_records(category=...)."""
    return AIMessage(
        content="",
        id=msg_id,
        tool_calls=[{
            "name": "count_records",
            "args": {"category": category},
            "id": f"call_{msg_id}",
            "type": "tool_call",
        }],
    )


def _fake_make_llm(bound_side_effect, plain_return=None):
    """Fake make_llm(): .bind_tools().invoke() is the agent; .invoke() is force_answer."""
    bound = MagicMock()
    bound.invoke.side_effect = bound_side_effect
    fake = MagicMock()
    fake.bind_tools.return_value = bound
    if plain_return is not None:
        fake.invoke.return_value = plain_return
    return fake


def _ai_with_tool_calls(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)


def test_out_of_scope_query_is_declined_without_calling_the_agent():
    with patch.object(graph_module, "classify_query",
                      return_value=MagicMock(route="out_of_scope")), \
         patch.object(graph_module, "make_llm") as mock_make_llm:
        result = build_graph().invoke(_initial_state("Write me a poem"), config=CONFIG)

    assert result["messages"][-1].content == DECLINE_MESSAGE
    mock_make_llm.assert_not_called()


def test_structured_query_runs_tool_then_returns_final_answer():
    final = AIMessage(content="We received 2992 refund-related requests.")
    fake = _fake_make_llm([_tool_call_message("ai_1"), final])

    with patch.object(graph_module, "classify_query",
                      return_value=MagicMock(route="structured")), \
         patch.object(graph_module, "make_llm", return_value=fake):
        result = build_graph().invoke(_initial_state("How many refunds?"), config=CONFIG)

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "2992" in tool_messages[0].content
    assert result["messages"][-1].content == final.content


def test_repeated_tool_call_triggers_force_answer():
    forced = AIMessage(content="Based on the data, there are 2992 refund requests.")
    # Agent asks for the same call twice; the guard should divert to force_answer.
    fake = _fake_make_llm(
        bound_side_effect=[_tool_call_message("ai_1"), _tool_call_message("ai_2")],
        plain_return=forced,
    )

    with patch.object(graph_module, "classify_query",
                      return_value=MagicMock(route="structured")), \
         patch.object(graph_module, "make_llm", return_value=fake):
        result = build_graph().invoke(_initial_state("How many refunds?"), config=CONFIG)

    assert result["messages"][-1].content == forced.content
    # Tool executed once; the redundant tool-call message was removed from history.
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 1
    assert _ai_with_tool_calls(result["messages"]) == 1


def test_episodic_memory_persists_across_graph_rebuilds(tmp_path):
    """SqliteSaver: a new build_graph() with the same db file resumes the prior conversation."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = str(tmp_path / "checkpoints.sqlite")
    session_id = "test_session"
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

    final = AIMessage(content="We received 2992 refund-related requests.")
    fake = _fake_make_llm([_tool_call_message("ai_1"), final])

    # First "process" — invoke once; the saver persists final state to db.
    with SqliteSaver.from_conn_string(db_path) as saver:
        with patch.object(graph_module, "classify_query",
                          return_value=MagicMock(route="structured")), \
             patch.object(graph_module, "make_llm", return_value=fake):
            graph = build_graph(checkpointer=saver)
            graph.invoke(_initial_state("How many refunds?"), config=config)

    # Second "process" — same db file, brand-new graph instance. No LLM patches
    # needed because we are not invoking, only reading state.
    with SqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(checkpointer=saver)
        state = graph.get_state(config)

    messages = state.values.get("messages", [])
    # The prior Human + AI(final) must have survived the restart.
    assert any(isinstance(m, HumanMessage) and "refunds" in str(m.content).lower()
               for m in messages)
    assert any(isinstance(m, AIMessage) and "2992" in str(m.content) for m in messages)


def test_personal_route_uses_profile_with_no_tools(tmp_path):
    """A personal query goes to the personal node, which reads the user's profile."""
    from agent import profile as profile_module

    forced = AIMessage(content="Your name is Alice; you are interested in REFUND data.")
    fake = _fake_make_llm(bound_side_effect=[], plain_return=forced)

    initial = {
        "messages": [HumanMessage(content="What do you remember about me?")],
        "route": "",
        "iterations": 0,
        "user_id": "alice",
    }

    with patch.object(profile_module, "_PROFILES_DIR", tmp_path), \
         patch.object(graph_module, "classify_query",
                      return_value=MagicMock(route="personal")), \
         patch.object(graph_module, "make_llm", return_value=fake):
        # Seed a profile so the personal node has something to use.
        (tmp_path / "alice.md").write_text("# Name\nAlice\n# Interests\n- REFUND data\n")
        result = build_graph().invoke(initial, config=CONFIG)

    # The agent (tool-binding) LLM was NEVER invoked — personal path uses no tools.
    fake.bind_tools.assert_not_called()
    # The final message is the personal node's answer.
    assert result["messages"][-1].content == forced.content
    # No ToolMessages produced.
    assert not [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_runaway_distinct_tool_calls_hit_fallback_after_max_iterations():
    counter = {"n": 0}

    def distinct_call(_messages):
        counter["n"] += 1
        # Different args each time -> never a duplicate -> the budget guard must catch it.
        return _tool_call_message(f"ai_{counter['n']}", category=f"UNKNOWN_{counter['n']}")

    fake = _fake_make_llm(bound_side_effect=distinct_call)

    with patch.object(graph_module, "classify_query",
                      return_value=MagicMock(route="structured")), \
         patch.object(graph_module, "make_llm", return_value=fake):
        result = build_graph().invoke(_initial_state("loop forever"), config=CONFIG)

    assert result["messages"][-1].content == FALLBACK_MESSAGE
    assert result["iterations"] == graph_module.MAX_ITERATIONS
