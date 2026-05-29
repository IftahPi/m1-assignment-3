"""Query router: classifies user queries into three labels."""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agent.schemas import RouteDecision
from nebius_client import make_llm, ROUTER_MODEL

ROUTER_PROMPT: str = """You are a query classifier for a customer-service data analyst agent.
The agent answers questions about the **Bitext Customer Service** dataset (26,872 rows).

The dataset has 11 categories: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
It has 27 intents: cancel_order, change_order, change_shipping_address, check_cancellation_fee, check_invoice, check_payment_methods, check_refund_policy, complaint, contact_customer_service, contact_human_agent, create_account, delete_account, delivery_options, delivery_period, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account, track_order, track_refund.

Your job is to classify a user query into ONE of three labels:

1. **structured** — questions with concrete, data-driven answers about the dataset (counts, lists, examples, distributions).
   - Examples: "How many refund requests did we get?", "What categories exist?", "Show me 5 SHIPPING examples", "distribution of intents in ACCOUNT"

2. **unstructured** — open-ended questions about the dataset needing summarization.
   - Examples: "Summarize the FEEDBACK category", "How do agents respond to complaints?"

3. **out_of_scope** — anything NOT about this customer-service dataset. These must be DECLINED and NEVER answered from general knowledge.
   - Examples: "Who won the 2024 Champions League?", "Write me a poem about customer service", "Who is the president of France?", "What's the best CRM software?"

FOLLOW-UP RULE — important. When the user message starts with "Recent conversation:" you are seeing prior turns, then the NEW user message to classify. The new message may be a short follow-up that depends on prior context (e.g. "what about refunds?", "what is the total of the two?", "show me 3 more", "what did I just ask?"). Such follow-ups inherit the topic of the prior conversation: if the conversation has been about this dataset, classify them as structured or unstructured (whichever fits the kind of answer they want), NOT out_of_scope. Reserve out_of_scope for messages whose own topic is not about the dataset.

Return your classification as a structured output with two fields:
- route: one of "structured", "unstructured", or "out_of_scope"
- reason: a short one-sentence explanation"""


def _format_recent_context(prior_messages: list[AnyMessage]) -> str:
    """Render the most recent user/agent turns as a short context block for the router."""
    recent = [m for m in prior_messages if isinstance(m, (HumanMessage, AIMessage))][-4:]
    if not recent:
        return ""
    lines: list[str] = []
    for message in recent:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        text = str(message.content).strip()
        if not text:
            continue  # skip AIMessages that only carried tool_calls with empty content
        lines.append(f"{role}: {text[:300]}")
    return "Recent conversation:\n" + "\n".join(lines) if lines else ""


def classify_query(
    query: str,
    prior_messages: list[AnyMessage] | None = None,
) -> RouteDecision:
    """Classify a user query into one of three routing labels.

    Args:
        query: The user's current natural-language question.
        prior_messages: Optional earlier turns of this conversation. When given, the
            router sees a short context block so it can correctly classify
            follow-ups like "what is the total?" or "what about refunds?".

    Returns:
        A RouteDecision with the route label and a brief reason.
    """
    llm = make_llm(ROUTER_MODEL, temperature=0.0)
    llm_with_output = llm.with_structured_output(RouteDecision)

    context = _format_recent_context(prior_messages or [])
    user_content = f"{context}\n\nNew user message to classify: {query}" if context else query

    decision = llm_with_output.invoke(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )

    return decision
