"""Shared state for the agent graph."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State threaded through the graph nodes.

    Attributes:
        messages: The running conversation; ``add_messages`` appends new turns.
        route: The router's classification ("structured" / "unstructured" / "out_of_scope").
        iterations: How many times the agent (LLM) node has run, for the max-iteration guard.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    iterations: int
