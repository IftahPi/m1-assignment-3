"""Shared state for the agent graph."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State threaded through the graph nodes.

    Attributes:
        messages: The running conversation; ``add_messages`` appends new turns.
        route: The router's classification ("structured" / "unstructured" /
            "out_of_scope" / "personal").
        iterations: How many times the agent (LLM) node has run, for the
            max-iteration guard.
        user_id: Which user this turn is for; the personal node uses it to load
            the per-user profile. Optional so older callers and tests that
            don't set it keep working.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    iterations: int
    user_id: NotRequired[str]
