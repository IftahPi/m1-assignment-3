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
    ToolMessage,
)
from langchain_core.tools import tool as tool_decorator
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent import profile as profile_module
from agent.router import classify_query
from agent.state import AgentState
from agent.tools import TOOLS
from nebius_client import GENERATOR_MODEL, ROUTER_MODEL, make_llm

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

PERSONAL_SYSTEM_PROMPT: str = """\
You are answering a question the user is asking ABOUT THEMSELVES (for example: \
"what do you remember about me?", "what's my name?", "what topics do I usually \
ask about?").

The current user_id is "{user_id}". You have ONE tool:
  get_personal_info(user_id: str) -> str
which returns this user's persistent profile as Markdown text.

Procedure: call get_personal_info("{user_id}") FIRST. Then base your answer \
ONLY on what that tool returns. If the returned text is empty or does not \
contain the information, say plainly: "I don't have that on file yet." Do \
NOT guess. Do NOT use general knowledge. Do NOT invent facts. No other tools \
are available.

Output goes to a plain-text terminal (a CLI). Do NOT use markdown: no \
**bold**, no tables, no #/## headers, no backticks. Plain prose only, with \
"- " bullets if you list things.
"""

_PERSONAL_NODE_MAX_STEPS: int = 3

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
    """Classify the latest user query into a route label, with prior turns as context.

    Passing the recent conversation lets the router correctly classify follow-ups
    like "what about refunds?" or "what is the total of the two?" — which look
    out-of-scope in isolation but inherit the topic of the prior exchange.
    """
    messages = state["messages"]
    decision = classify_query(_last_human_text(messages), prior_messages=messages[:-1])
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


def _personal_node(state: AgentState) -> dict:
    """Answer a personal question using ONLY the persistent user profile.

    Architecture: the personal node binds exactly ONE tool — a closure-scoped
    ``get_personal_info(user_id)`` locked to the current user_id. The data
    tools (``count_records``, ``get_examples``, …) are unreachable here, so
    the model cannot answer a personal question from dataset facts or world
    knowledge — it either reads the profile via the tool or admits it doesn't
    know. A small ReAct mini-loop runs the tool call + final answer in this
    one node, then the graph goes to END.
    """
    current_user_id = state.get("user_id") or "default"

    @tool_decorator
    def get_personal_info(user_id: str) -> str:
        """Return the persistent profile Markdown for the given user.

        The user_id you pass MUST be the current session's user — any other id
        is refused. The profile may be empty (a brand-new user); do not invent
        content when it is.
        """
        if user_id != current_user_id:
            return (
                f"(Access restricted: this tool only returns the current user's profile "
                f"('{current_user_id}'); you asked for '{user_id}').)"
            )
        return profile_module.get_personal_info(current_user_id) or "(empty profile — nothing remembered yet)"

    personal_tools = [get_personal_info]
    # The personal task is "call one tool, paraphrase its short result" — squarely the
    # cheap router model's strength. No need for the 120B generator here.
    llm = make_llm(ROUTER_MODEL, temperature=0.2).bind_tools(personal_tools)

    new_messages: list[AnyMessage] = []
    history: list[AnyMessage] = [
        SystemMessage(content=PERSONAL_SYSTEM_PROMPT.format(user_id=current_user_id)),
        *state["messages"],
    ]

    for _ in range(_PERSONAL_NODE_MAX_STEPS):
        response = llm.invoke(history)
        new_messages.append(response)
        history.append(response)
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            result = get_personal_info.invoke(call["args"])
            tm = ToolMessage(content=result, tool_call_id=call["id"])
            new_messages.append(tm)
            history.append(tm)

    return {"messages": new_messages}


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
    """Route by the router's label: out_of_scope → decline, personal → personal_node, else agent."""
    route = state["route"]
    if route == "out_of_scope":
        return "decline"
    if route == "personal":
        return "personal"
    return "agent"


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
    builder.add_node("personal", _personal_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {"decline": "decline", "agent": "agent", "personal": "personal"},
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
    builder.add_edge("personal", END)
    return builder.compile(checkpointer=checkpointer)
