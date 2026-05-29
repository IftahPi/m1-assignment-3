"""Pydantic schemas for the agent.

The category/intent fields are typed as Literals of the dataset's actual values,
so the LLM cannot accidentally pass an intent name as a category (or vice versa).
A BeforeValidator normalises case before the Literal check, so "shipping",
"Shipping", and "SHIPPING" all validate against the same SHIPPING category;
similarly "Get_Refund" -> "get_refund". When the slot is wrong (e.g. the model
passes intent='SHIPPING'), Pydantic raises a clear ValidationError whose
message lists the valid intents — the agent then retries with the right slot.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

CategoryName = Literal[
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]

IntentName = Literal[
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
]


def _to_upper(value: object) -> object:
    """Uppercase a string in place; pass anything else through unchanged."""
    return value.upper() if isinstance(value, str) else value


def _to_lower(value: object) -> object:
    """Lowercase a string in place; pass anything else through unchanged."""
    return value.lower() if isinstance(value, str) else value


# Case-normalised, list-constrained aliases. Use these for the agent's filter fields.
NormalizedCategory = Annotated[CategoryName, BeforeValidator(_to_upper)]
NormalizedIntent = Annotated[IntentName, BeforeValidator(_to_lower)]


class RouteDecision(BaseModel):
    """Result of query routing classification."""

    route: Literal["structured", "unstructured", "out_of_scope", "personal"] = Field(
        description="The classification of the user query: structured (data-driven answers), "
        "unstructured (summarization), out_of_scope (not about the dataset), or personal "
        "(about the user themselves — answered from the persistent user profile)."
    )
    reason: str = Field(
        description="One short sentence explaining why the query was classified this way."
    )


class ListIntentsInput(BaseModel):
    """Input schema for the list_intents tool."""

    category: NormalizedCategory | None = Field(
        default=None,
        description="Optional category to scope intents to (e.g. 'REFUND'). Omit for all intents.",
    )


class CountRecordsInput(BaseModel):
    """Input schema for the count_records tool."""

    category: NormalizedCategory | None = Field(
        default=None,
        description="Optional category filter, e.g. 'REFUND'. Must be one of the dataset's 11 categories.",
    )
    intent: NormalizedIntent | None = Field(
        default=None,
        description="Optional intent filter, e.g. 'get_refund'. Must be one of the dataset's 27 intents.",
    )
    text_contains: str | None = Field(
        default=None,
        description="Optional case-insensitive substring to match in the customer message.",
    )


class GetExamplesInput(BaseModel):
    """Input schema for the get_examples tool."""

    category: NormalizedCategory | None = Field(
        default=None,
        description="Optional category filter. Must be one of the dataset's 11 categories.",
    )
    intent: NormalizedIntent | None = Field(
        default=None,
        description="Optional intent filter. Must be one of the dataset's 27 intents.",
    )
    text_contains: str | None = Field(
        default=None, description="Optional case-insensitive substring filter on the message."
    )
    n: int = Field(default=5, ge=1, le=50, description="How many examples to return (1-50).")


class IntentDistributionInput(BaseModel):
    """Input schema for the intent_distribution tool."""

    category: NormalizedCategory | None = Field(
        default=None,
        description="Optional category to scope the distribution to. Omit for the whole dataset.",
    )


class SearchExamplesInput(BaseModel):
    """Input schema for the search_examples tool."""

    query: str = Field(
        description="Free-text phrase to search for inside customer messages, e.g. 'money back'."
    )
    n: int = Field(default=5, ge=1, le=50, description="How many matching examples to return (1-50).")
