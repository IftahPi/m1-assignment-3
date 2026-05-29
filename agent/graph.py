"""The agent graph: a routed ReAct loop with a max-iteration guard.

Flow:  START -> router -> (decline | agent)
       agent  <-> tools           (ReAct loop while the LLM requests tools)
       agent  -> fallback         (if the iteration budget is exhausted)
       decline / fallback / final answer -> END
"""

import json

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.router import classify_query
from agent.state import AgentState
from agent.tools import TOOLS
from nebius_client import GENERATOR_MODEL, make_llm

MAX_ITERATIONS: int = 12

AGENT_SYSTEM_PROMPT: str = """You are a data analyst for the Bitext Customer Service dataset \
(26,872 rows of customer support messages paired with agent responses).

Each row has: category (one of 11 high-level groups), intent (one of 27 fine-grained intents), \
instruction (the customer's message), and response (the agent's reply).
Categories: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
Intents: cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, \
check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, \
create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, \
get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, \
review, set_up_shipping_address, switch_account, track_order, track_refund.

DECIDING CATEGORY vs INTENT: look the word up in the two lists above — that is the only correct test. \
If the word appears in the categories list (e.g. SHIPPING, REFUND, ACCOUNT) put it in the `category` \
parameter; if it appears in the intents list (e.g. get_refund, change_shipping_address) put it in the \
`intent` parameter. The user's letter case can vary ("shipping", "Shipping", "SHIPPING" all refer to \
the same category) — match by membership, not by capitalization. The user can also misuse the words: \
phrases like "the SHIPPING intent" still mean the SHIPPING *category*, because SHIPPING is in the \
categories list, not the intents list.

Always use the provided tools to get facts — never invent numbers or examples. Map the user's wording \
to the right filter, e.g. "refund requests" -> category 'REFUND' (or intent 'get_refund'); "complaints" \
-> intent 'complaint'; a phrase in the user's own words -> search_examples.

Rules for using tools:
- Before EVERY tool call, write ONE short sentence in your reply content explaining what you are about \
to look up and why. Keep it brief — one sentence. Put it in the assistant message content, alongside \
the tool call itself.
- Never call the same tool twice with the same arguments. One call returns all the data you need from it.
- In particular, call get_examples or search_examples EXACTLY ONCE per request, then present those rows.
- As soon as a tool has returned the data, STOP calling tools and write the final answer.
- If a question needs several different facts (e.g. two counts to add up), make those few distinct calls, \
then combine the results.
- If a tool returns an EMPTY result ([] or 0), do not stop and do not describe what the tool would do. \
Most often the user named a category but you used the intent slot (or vice-versa) — retry ONCE with the \
swapped slot (category <-> intent). If it's still empty, tell the user plainly that nothing matched.

Rules for the final answer:
- Always include the concrete result itself — the number, the list, or the example messages — not just a \
description of what you did. Briefly note which filter you assumed.
- When showing examples, list each one's customer message (and a short snippet of the agent's reply).
- For open-ended/summarization questions, fetch representative rows with get_examples or search_examples, \
then summarize them yourself. Keep answers concise and grounded in the tool results.

OUTPUT FORMAT — plain text only. Your reply is shown in a plain-text terminal (a CLI), NOT a markdown \
viewer. NEVER use markdown formatting: no **bold** or *italics*, no # or ## headers, no tables with | \
pipes, no backticks, no markdown links. Write numbers plainly ("1,000" not "**1,000**"). For lists use \
plain bullets ("- " or "• ") or "1. / 2. / 3.". For distributions and key-value summaries, write one \
"name: count" per line. Plain prose only."""

DECLINE_MESSAGE: str = (
    "I'm a data analyst for the Bitext customer-service dataset, so I can only answer questions "
    "about that data — categories, intents, counts, examples, and summaries. I can't help with "
    "that request, but feel free to ask me about the dataset."
)

FALLBACK_MESSAGE: str = (
    "I wasn't able to work that out within my step budget. Could you rephrase or narrow the "
    "question — for example, name a specific category or intent?"
)

_FORCE_ANSWER_INSTRUCTION: str = (
    "You have already gathered the necessary data from the tools above. Do not call any more "
    "tools. Answer the user's question now using the results you already have."
)


def _tool_call_signature(call: dict) -> str:
    """A stable identity for a tool call: its name plus its sorted arguments."""
    return call["name"] + ":" + json.dumps(call.get("args", {}), sort_keys=True, default=str)


def _prior_tool_signatures(messages: list[AnyMessage]) -> set[str]:
    """Signatures of every tool call already issued, excluding the final message."""
    signatures: set[str] = set()
    for message in messages[:-1]:
        if isinstance(message, AIMessage) and message.tool_calls:
            signatures.update(_tool_call_signature(call) for call in message.tool_calls)
    return signatures


def _last_human_text(messages: list[AnyMessage]) -> str:
    """Return the text of the most recent human message, or '' if none."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _router_node(state: AgentState) -> dict:
    """Classify the latest user query into a route label."""
    decision = classify_query(_last_human_text(state["messages"]))
    return {"route": decision.route}


def _agent_node(state: AgentState) -> dict:
    """Run the generator LLM (bound to tools) over the conversation."""
    llm = make_llm(GENERATOR_MODEL, temperature=0.2).bind_tools(TOOLS)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}


def _decline_node(state: AgentState) -> dict:
    """Politely decline an out-of-scope query."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


def _fallback_node(state: AgentState) -> dict:
    """Return a graceful message when the iteration budget is exhausted."""
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def _force_answer_node(state: AgentState) -> dict:
    """Force a tool-free final answer when the agent repeats a tool call it already made.

    Drops the redundant (unexecuted) tool-call message and asks the generator, with no
    tools bound, to answer from the results already gathered.
    """
    redundant_call = state["messages"][-1]
    llm = make_llm(GENERATOR_MODEL, temperature=0.2)
    prompt = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        *state["messages"][:-1],
        SystemMessage(content=_FORCE_ANSWER_INSTRUCTION),
    ]
    response = llm.invoke(prompt)
    iterations = state.get("iterations", 0) + 1
    if redundant_call.id is not None:
        return {"messages": [RemoveMessage(id=redundant_call.id), response], "iterations": iterations}
    return {"messages": [response], "iterations": iterations}


def _route_after_router(state: AgentState) -> str:
    """Send out-of-scope queries to decline; everything else to the agent."""
    return "decline" if state["route"] == "out_of_scope" else "agent"


def _route_after_agent(state: AgentState) -> str:
    """Route the agent's output: finish, loop to tools, force an answer, or fall back.

    - No tool calls -> the agent answered, so END.
    - Iteration budget exhausted -> fallback.
    - All requested tool calls were already made (no progress) -> force an answer.
    - Otherwise -> run the requested tools.
    """
    last = state["messages"][-1]
    wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not wants_tools:
        return END
    if state["iterations"] >= MAX_ITERATIONS:
        return "fallback"
    requested = {_tool_call_signature(call) for call in last.tool_calls}
    if requested and requested <= _prior_tool_signatures(state["messages"]):
        return "force_answer"
    return "tools"


def build_graph(checkpointer=None):
    """Build and compile the agent graph. ``checkpointer`` enables persistence (later tasks)."""
    builder = StateGraph(AgentState)
    builder.add_node("router", _router_node)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("decline", _decline_node)
    builder.add_node("fallback", _fallback_node)
    builder.add_node("force_answer", _force_answer_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", _route_after_router, {"decline": "decline", "agent": "agent"}
    )
    builder.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "fallback": "fallback", "force_answer": "force_answer", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("decline", END)
    builder.add_edge("fallback", END)
    builder.add_edge("force_answer", END)
    return builder.compile(checkpointer=checkpointer)
