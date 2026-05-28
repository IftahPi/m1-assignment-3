"""Query router: classifies user queries into three labels."""

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

Return your classification as a structured output with two fields:
- route: one of "structured", "unstructured", or "out_of_scope"
- reason: a short one-sentence explanation"""


def classify_query(query: str) -> RouteDecision:
    """Classify a user query into one of three routing labels.

    Args:
        query: The user's natural-language question.

    Returns:
        A RouteDecision with the route label and a brief reason.
    """
    llm = make_llm(ROUTER_MODEL, temperature=0.0)
    llm_with_output = llm.with_structured_output(RouteDecision)

    decision = llm_with_output.invoke(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query},
        ]
    )

    return decision
