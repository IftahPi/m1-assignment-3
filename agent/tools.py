"""LangChain tools the agent can call.

Each tool is a thin, well-described wrapper over a pure function in
``dataset.analytics``. The descriptions matter as much as the logic: they tell
the LLM *when* to reach for each tool. The dataset is loaded once and cached.
"""

from langchain_core.tools import tool

from agent.schemas import (
    CountRecordsInput,
    GetExamplesInput,
    IntentDistributionInput,
    ListIntentsInput,
    SearchExamplesInput,
)
from dataset import analytics
from dataset.loader import load_dataframe


@tool
def list_categories() -> list[str]:
    """List every high-level category in the dataset (e.g. REFUND, ORDER, SHIPPING).

    Use this to answer 'what categories exist?' or before filtering, to confirm
    the available category names.
    """
    return analytics.list_categories(load_dataframe())


@tool(args_schema=ListIntentsInput)
def list_intents(category: str | None = None) -> list[str]:
    """List the fine-grained intents, optionally within one category.

    Use this to see which intents exist (e.g. the intents inside the ACCOUNT
    category) or to map a vague request to a concrete intent name.
    """
    return analytics.list_intents(load_dataframe(), category=category)


@tool(args_schema=CountRecordsInput)
def count_records(
    category: str | None = None,
    intent: str | None = None,
    text_contains: str | None = None,
) -> int:
    """Count how many dataset rows match the given filters (category, intent, text).

    Use this for any 'how many…' question. Filters combine with AND; omit a
    filter to ignore it. Example: count refund requests with category='REFUND'.
    """
    return analytics.count_records(
        load_dataframe(), category=category, intent=intent, text_contains=text_contains
    )


@tool(args_schema=GetExamplesInput)
def get_examples(
    category: str | None = None,
    intent: str | None = None,
    text_contains: str | None = None,
    n: int = 5,
) -> list[dict[str, str]]:
    """Return up to n example rows (customer message + agent response) matching filters.

    Use this to 'show me examples of…' a category/intent, and also to gather raw
    rows to summarize for open-ended questions about how customers or agents phrase things.
    """
    return analytics.get_examples(
        load_dataframe(), category=category, intent=intent, text_contains=text_contains, n=n
    )


@tool(args_schema=IntentDistributionInput)
def intent_distribution(category: str | None = None) -> dict[str, int]:
    """Return a mapping of intent -> row count, optionally scoped to one category.

    Use this for 'what is the distribution of intents in X?' questions.
    """
    return analytics.intent_distribution(load_dataframe(), category=category)


@tool(args_schema=SearchExamplesInput)
def search_examples(query: str, n: int = 5) -> list[dict[str, str]]:
    """Find up to n examples whose customer message contains a free-text phrase.

    Use this when the user describes something in their own words rather than by
    category/intent, e.g. 'people wanting their money back'.
    """
    return analytics.search_examples(load_dataframe(), query=query, n=n)


TOOLS: list = [
    list_categories,
    list_intents,
    count_records,
    get_examples,
    intent_distribution,
    search_examples,
]
