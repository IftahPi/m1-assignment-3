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
