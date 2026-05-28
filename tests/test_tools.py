"""Integration tests for the agent tools against the real cached dataset.

These read the local CSV (no network, no LLM). They assert known dataset facts
and that every tool carries a name + description (graded requirement).
"""

from agent import tools
from agent.tools import (
    count_records,
    get_examples,
    intent_distribution,
    list_categories,
    list_intents,
    search_examples,
)

ALL_CATEGORIES = {
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
}


def test_list_categories_returns_all_eleven():
    assert set(list_categories.invoke({})) == ALL_CATEGORIES


def test_list_intents_for_account_category():
    assert set(list_intents.invoke({"category": "ACCOUNT"})) == {
        "create_account", "delete_account", "edit_account",
        "recover_password", "registration_problems", "switch_account",
    }


def test_count_records_refund_category_known_value():
    assert count_records.invoke({"category": "REFUND"}) == 2992


def test_get_examples_returns_requested_count_and_matches_filter():
    examples = get_examples.invoke({"category": "SHIPPING", "n": 3})
    assert len(examples) == 3
    assert all(e["category"] == "SHIPPING" for e in examples)


def test_intent_distribution_refund_sums_to_category_total():
    dist = intent_distribution.invoke({"category": "REFUND"})
    assert set(dist) == {"check_refund_policy", "get_refund", "track_refund"}
    assert sum(dist.values()) == 2992


def test_search_examples_matches_query_text():
    results = search_examples.invoke({"query": "money back", "n": 5})
    assert len(results) >= 1
    assert all("money back" in r["instruction"].lower() for r in results)


def test_every_tool_has_name_and_description():
    for tool in tools.TOOLS:
        assert tool.name
        assert tool.description and len(tool.description) > 20
