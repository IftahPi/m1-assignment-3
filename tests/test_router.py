"""Test the query router."""

from unittest.mock import MagicMock, patch

from agent.router import classify_query, ROUTER_PROMPT
from agent.schemas import RouteDecision
from nebius_client import ROUTER_MODEL


def test_classify_query_structured():
    """Test that structured queries are classified correctly."""
    mock_llm = MagicMock()
    mock_response = RouteDecision(
        route="structured",
        reason="User is asking for a count of records."
    )
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("agent.router.make_llm", return_value=mock_llm) as mock_make_llm:
        result = classify_query("How many refund requests did we get?")

    assert result.route == "structured"
    assert result.reason == "User is asking for a count of records."
    mock_llm.with_structured_output.assert_called_once()
    mock_make_llm.assert_called_once_with(ROUTER_MODEL, temperature=0.0)


def test_classify_query_unstructured():
    """Test that unstructured queries are classified correctly."""
    mock_llm = MagicMock()
    mock_response = RouteDecision(
        route="unstructured",
        reason="User wants a summary of complaints."
    )
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("agent.router.make_llm", return_value=mock_llm):
        result = classify_query("Summarize the FEEDBACK category")

    assert result.route == "unstructured"
    assert result.reason == "User wants a summary of complaints."


def test_classify_query_out_of_scope():
    """Test that out-of-scope queries are classified correctly."""
    mock_llm = MagicMock()
    mock_response = RouteDecision(
        route="out_of_scope",
        reason="Not related to the customer service dataset."
    )
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("agent.router.make_llm", return_value=mock_llm):
        result = classify_query("Who won the 2024 Champions League?")

    assert result.route == "out_of_scope"
    assert result.reason == "Not related to the customer service dataset."


def test_router_prompt_mentions_labels():
    """Test that the router prompt mentions all four route labels."""
    assert "structured" in ROUTER_PROMPT
    assert "unstructured" in ROUTER_PROMPT
    assert "out_of_scope" in ROUTER_PROMPT
    assert "personal" in ROUTER_PROMPT
    assert "remember about me" in ROUTER_PROMPT.lower()


def test_router_prompt_mentions_decline_rule():
    """The prompt must instruct that out-of-scope queries are declined, not answered."""
    prompt = ROUTER_PROMPT.lower()
    assert "out_of_scope" in prompt
    assert "declined" in prompt or "decline" in prompt
    assert "never answered from general knowledge" in prompt


def test_router_prompt_includes_follow_up_rule():
    """The prompt must tell the router how to treat follow-up questions in a session."""
    prompt = ROUTER_PROMPT.lower()
    assert "follow-up" in prompt
    assert "recent conversation" in prompt


def test_classify_query_passes_recent_history_to_the_llm():
    """When prior_messages are given, the router invoke sees a 'Recent conversation:' block."""
    from langchain_core.messages import AIMessage, HumanMessage

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = RouteDecision(
        route="structured", reason="follow-up about counts"
    )

    history = [
        HumanMessage(content="How many complaints did we get?"),
        AIMessage(content="There are 1,000 complaints."),
    ]

    with patch("agent.router.make_llm", return_value=mock_llm):
        classify_query("What about refunds?", prior_messages=history)

    # The structured-output LLM was invoked with a 2-message list whose user
    # content carries both the recent context and the new question.
    call_args = mock_llm.with_structured_output.return_value.invoke.call_args
    msgs = call_args[0][0]
    user_content = msgs[1]["content"]
    assert "Recent conversation:" in user_content
    assert "complaints" in user_content.lower()
    assert "What about refunds?" in user_content


def test_agent_system_prompt_forbids_markdown_for_cli_output():
    """The agent prompt must tell the model its output is plain-text CLI, no markdown."""
    from agent.graph import AGENT_SYSTEM_PROMPT

    text = AGENT_SYSTEM_PROMPT.lower()
    assert "plain text" in text or "plain-text" in text
    assert "markdown" in text


ALL_INTENTS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
]


def test_router_prompt_lists_all_27_intents():
    """The prompt must enumerate every one of the 27 dataset intents."""
    assert len(ALL_INTENTS) == 27
    missing = [intent for intent in ALL_INTENTS if intent not in ROUTER_PROMPT]
    assert not missing, f"intents absent from ROUTER_PROMPT: {missing}"


def test_make_llm_called_with_router_model():
    """Test that classify_query calls make_llm with ROUTER_MODEL and temperature=0.0."""
    mock_llm = MagicMock()
    mock_response = RouteDecision(
        route="structured",
        reason="Test reason."
    )
    mock_llm.with_structured_output.return_value.invoke.return_value = mock_response

    with patch("agent.router.make_llm", return_value=mock_llm) as mock_make_llm:
        classify_query("Test query")

    mock_make_llm.assert_called_once_with(ROUTER_MODEL, temperature=0.0)
