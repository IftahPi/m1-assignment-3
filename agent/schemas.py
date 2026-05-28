"""Pydantic schemas for the agent."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Result of query routing classification."""

    route: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="The classification of the user query: structured (data-driven answers), "
        "unstructured (summarization), or out_of_scope (not about the dataset)."
    )
    reason: str = Field(
        description="One short sentence explaining why the query was classified this way."
    )


class ListIntentsInput(BaseModel):
    """Input schema for the list_intents tool."""

    category: str | None = Field(
        default=None,
        description="Optional category to scope intents to (e.g. 'REFUND'). Omit for all intents.",
    )


class CountRecordsInput(BaseModel):
    """Input schema for the count_records tool."""

    category: str | None = Field(
        default=None, description="Optional category filter, e.g. 'REFUND' (case-insensitive)."
    )
    intent: str | None = Field(
        default=None, description="Optional intent filter, e.g. 'get_refund' (case-insensitive)."
    )
    text_contains: str | None = Field(
        default=None,
        description="Optional case-insensitive substring to match in the customer message.",
    )


class GetExamplesInput(BaseModel):
    """Input schema for the get_examples tool."""

    category: str | None = Field(default=None, description="Optional category filter.")
    intent: str | None = Field(default=None, description="Optional intent filter.")
    text_contains: str | None = Field(
        default=None, description="Optional case-insensitive substring filter on the message."
    )
    n: int = Field(default=5, ge=1, le=50, description="How many examples to return (1-50).")


class IntentDistributionInput(BaseModel):
    """Input schema for the intent_distribution tool."""

    category: str | None = Field(
        default=None,
        description="Optional category to scope the distribution to. Omit for the whole dataset.",
    )


class SearchExamplesInput(BaseModel):
    """Input schema for the search_examples tool."""

    query: str = Field(
        description="Free-text phrase to search for inside customer messages, e.g. 'money back'."
    )
    n: int = Field(default=5, ge=1, le=50, description="How many matching examples to return (1-50).")
