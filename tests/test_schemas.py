"""Schema-level validation for tool inputs: Literal types + case normalization."""

import pytest
from pydantic import ValidationError

from agent.schemas import (
    CountRecordsInput,
    GetExamplesInput,
    ListIntentsInput,
    RouteDecision,
)


def test_category_case_is_normalized_to_upper():
    assert ListIntentsInput(category="shipping").category == "SHIPPING"
    assert ListIntentsInput(category="Shipping").category == "SHIPPING"
    assert ListIntentsInput(category="SHIPPING").category == "SHIPPING"


def test_intent_case_is_normalized_to_lower():
    assert CountRecordsInput(intent="GET_REFUND").intent == "get_refund"
    assert CountRecordsInput(intent="Get_Refund").intent == "get_refund"


def test_unknown_category_raises_validation_error():
    with pytest.raises(ValidationError):
        CountRecordsInput(category="NOPE")


def test_unknown_intent_raises_validation_error():
    with pytest.raises(ValidationError):
        CountRecordsInput(intent="nope_intent")


def test_category_name_in_intent_slot_is_rejected():
    """The whole point of the schema fix: SHIPPING is a category, not an intent."""
    with pytest.raises(ValidationError):
        CountRecordsInput(intent="SHIPPING")


def test_intent_name_in_category_slot_is_rejected():
    with pytest.raises(ValidationError):
        CountRecordsInput(category="get_refund")


def test_both_slots_valid_together():
    inp = GetExamplesInput(category="refund", intent="GET_REFUND", n=3)
    assert inp.category == "REFUND"
    assert inp.intent == "get_refund"
    assert inp.n == 3


def test_route_decision_still_validates():
    decision = RouteDecision(route="structured", reason="counts")
    assert decision.route == "structured"
