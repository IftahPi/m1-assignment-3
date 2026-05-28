"""Unit tests for the pure dataset analytics functions (no LLM, no I/O)."""

import pandas as pd

from dataset import analytics


def _make_df() -> pd.DataFrame:
    """A tiny, fully-known fixture frame mirroring the real schema."""
    rows = [
        ("B", "I want a refund please", "REFUND", "get_refund", "Sure, here is how."),
        ("B", "where is my refund", "REFUND", "track_refund", "Let me check."),
        ("B", "I want my money back", "REFUND", "get_refund", "Of course."),
        ("B", "cancel my order now", "ORDER", "cancel_order", "Cancelling now."),
        ("B", "track my order", "ORDER", "track_order", "Tracking it."),
        ("B", "file a complaint", "FEEDBACK", "complaint", "Sorry to hear that."),
    ]
    return pd.DataFrame(rows, columns=["flags", "instruction", "category", "intent", "response"])


def test_list_categories_sorted_and_distinct():
    assert analytics.list_categories(_make_df()) == ["FEEDBACK", "ORDER", "REFUND"]


def test_list_intents_all():
    assert analytics.list_intents(_make_df()) == [
        "cancel_order", "complaint", "get_refund", "track_order", "track_refund",
    ]


def test_list_intents_scoped_to_category():
    assert analytics.list_intents(_make_df(), category="REFUND") == ["get_refund", "track_refund"]


def test_list_intents_category_is_case_insensitive():
    df = _make_df()
    assert analytics.list_intents(df, category="refund") == analytics.list_intents(df, category="REFUND")


def test_count_records_by_category():
    assert analytics.count_records(_make_df(), category="REFUND") == 3


def test_count_records_by_intent():
    assert analytics.count_records(_make_df(), intent="get_refund") == 2


def test_count_records_combined_filters_are_anded():
    assert analytics.count_records(_make_df(), category="REFUND", intent="get_refund") == 2


def test_count_records_text_contains():
    assert analytics.count_records(_make_df(), text_contains="money back") == 1


def test_count_records_text_contains_is_case_insensitive():
    assert analytics.count_records(_make_df(), text_contains="MONEY BACK") == 1


def test_count_records_unknown_value_is_zero():
    assert analytics.count_records(_make_df(), category="NOPE") == 0


def test_get_examples_caps_at_n():
    examples = analytics.get_examples(_make_df(), category="REFUND", n=2)
    assert len(examples) == 2


def test_get_examples_caps_at_available_rows():
    examples = analytics.get_examples(_make_df(), category="REFUND", n=10)
    assert len(examples) == 3


def test_get_examples_rows_match_filter_and_have_expected_keys():
    examples = analytics.get_examples(_make_df(), category="REFUND", n=10)
    assert all(e["category"] == "REFUND" for e in examples)
    assert all(set(e) == {"category", "intent", "instruction", "response"} for e in examples)


def test_get_examples_unknown_value_is_empty():
    assert analytics.get_examples(_make_df(), category="NOPE") == []


def test_intent_distribution_scoped():
    assert analytics.intent_distribution(_make_df(), category="REFUND") == {
        "get_refund": 2,
        "track_refund": 1,
    }


def test_intent_distribution_values_are_plain_ints():
    dist = analytics.intent_distribution(_make_df())
    assert dist["get_refund"] == 2
    assert all(isinstance(v, int) for v in dist.values())


def test_search_examples_finds_substring_case_insensitive():
    results = analytics.search_examples(_make_df(), query="MONEY")
    assert len(results) == 1
    assert "money back" in results[0]["instruction"].lower()


def test_search_examples_caps_at_n():
    results = analytics.search_examples(_make_df(), query="refund", n=1)
    assert len(results) == 1
