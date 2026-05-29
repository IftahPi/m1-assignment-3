"""Tests for the per-user semantic-memory profile layer."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import profile as profile_module
from agent.profile import (
    SUMMARY_SYSTEM_PROMPT,
    _format_transcript,
    get_personal_info,
    get_personal_storage_file,
    save_profile,
    summarize_session,
)


def test_get_personal_storage_file_returns_user_specific_path(tmp_path):
    """The path scheme lives in ONE function and is per-user."""
    with patch.object(profile_module, "_PROFILES_DIR", tmp_path):
        assert get_personal_storage_file("alice") == tmp_path / "alice.md"
        assert get_personal_storage_file("bob") == tmp_path / "bob.md"


def test_summary_prompt_includes_contradiction_rule():
    """The summary must replace contradicted facts, not keep both."""
    text = SUMMARY_SYSTEM_PROMPT.lower()
    assert "contradict" in text
    assert "remove" in text or "replace" in text


def test_get_personal_info_returns_empty_string_when_no_file(tmp_path):
    with patch.object(profile_module, "_PROFILES_DIR", tmp_path):
        assert get_personal_info("nobody") == ""


def test_save_then_load_round_trips(tmp_path):
    blob = "# Name\nAlice\n\n# Interests\n- REFUND data\n"
    with patch.object(profile_module, "_PROFILES_DIR", tmp_path):
        save_profile("alice", blob)
        assert get_personal_info("alice") == blob
        assert (tmp_path / "alice.md").exists()


def test_save_profile_creates_directory_if_missing(tmp_path):
    nested = tmp_path / "does_not_exist_yet"
    with patch.object(profile_module, "_PROFILES_DIR", nested):
        save_profile("alice", "hi")
        assert (nested / "alice.md").read_text() == "hi"


def test_format_transcript_skips_tool_messages_and_empty_ai_content():
    messages = [
        HumanMessage(content="How many refunds?"),
        AIMessage(content="", tool_calls=[{"name": "count_records",
                                            "args": {"category": "REFUND"},
                                            "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="2992", tool_call_id="c1"),
        AIMessage(content="There are 2,992 refund requests."),
    ]
    transcript = _format_transcript(messages)
    assert "user: How many refunds?" in transcript
    assert "assistant: There are 2,992 refund requests." in transcript
    # ToolMessages and empty tool-call AIMessages must not appear.
    assert "count_records" not in transcript
    assert "2992\n" not in transcript


def test_summarize_session_calls_llm_with_existing_profile_and_transcript():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content="# Name\nAlice\n\n# Interests\n- REFUND data"
    )

    messages = [
        HumanMessage(content="My name is Alice and I'm interested in REFUND data."),
        AIMessage(content="Got it, Alice."),
    ]
    result = summarize_session(messages, prior_profile="", llm=llm)

    assert result == "# Name\nAlice\n\n# Interests\n- REFUND data"

    sent = llm.invoke.call_args[0][0]
    assert sent[0] == {"role": "system", "content": SUMMARY_SYSTEM_PROMPT}
    user_text = sent[1]["content"]
    assert "EXISTING PROFILE" in user_text
    assert "(none)" in user_text  # empty prior profile rendered explicitly
    assert "NEW SESSION TRANSCRIPT" in user_text
    assert "user: My name is Alice" in user_text


def test_summarize_session_includes_prior_profile_when_present():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="merged profile")

    summarize_session(
        messages=[HumanMessage(content="hi")],
        prior_profile="# Name\nAlice",
        llm=llm,
    )

    user_text = llm.invoke.call_args[0][0][1]["content"]
    assert "EXISTING PROFILE:\n# Name\nAlice" in user_text


def test_summary_prompt_says_not_a_transcript():
    """The system prompt makes it explicit the profile is distilled facts, not a replay."""
    text = SUMMARY_SYSTEM_PROMPT.lower()
    assert "distilled facts" in text
    assert "not a transcript" in text or "not a recap" in text
