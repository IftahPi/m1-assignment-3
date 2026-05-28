"""Pure analytical operations over the Bitext dataset DataFrame.

Every function takes the DataFrame as its first argument and returns plain,
JSON-serializable Python values. No LLM, no I/O — so these are unit-tested
directly against a small fixture frame, and are reused by both the agent's
tools and the MCP server.

Category matching is case-insensitive; intent matching is case-insensitive;
``text_contains`` is a case-insensitive substring match over the customer
``instruction`` text. Unknown filter values simply yield empty results.
"""

import pandas as pd

_EXAMPLE_COLUMNS: list[str] = ["category", "intent", "instruction", "response"]


def _filter(
    df: pd.DataFrame,
    category: str | None = None,
    intent: str | None = None,
    text_contains: str | None = None,
) -> pd.DataFrame:
    """Return the rows matching all provided filters (AND-combined)."""
    mask = pd.Series(True, index=df.index)
    if category is not None:
        mask &= df["category"].str.upper() == category.strip().upper()
    if intent is not None:
        mask &= df["intent"].str.lower() == intent.strip().lower()
    if text_contains is not None:
        mask &= df["instruction"].str.contains(text_contains, case=False, na=False, regex=False)
    return df[mask]


def _rows_to_dicts(rows: pd.DataFrame) -> list[dict[str, str]]:
    """Convert example rows to a list of plain string dicts."""
    return [
        {column: str(row[column]) for column in _EXAMPLE_COLUMNS}
        for _, row in rows.iterrows()
    ]


def list_categories(df: pd.DataFrame) -> list[str]:
    """Return all distinct categories in the dataset, sorted alphabetically."""
    return sorted(df["category"].unique().tolist())


def list_intents(df: pd.DataFrame, category: str | None = None) -> list[str]:
    """Return distinct intents, optionally scoped to a single category."""
    subset = _filter(df, category=category)
    return sorted(subset["intent"].unique().tolist())


def count_records(
    df: pd.DataFrame,
    category: str | None = None,
    intent: str | None = None,
    text_contains: str | None = None,
) -> int:
    """Return how many rows match the given category/intent/text filters."""
    return int(len(_filter(df, category, intent, text_contains)))


def get_examples(
    df: pd.DataFrame,
    category: str | None = None,
    intent: str | None = None,
    text_contains: str | None = None,
    n: int = 5,
) -> list[dict[str, str]]:
    """Return up to ``n`` random example rows matching the filters."""
    subset = _filter(df, category, intent, text_contains)
    if subset.empty:
        return []
    sample = subset.sample(min(n, len(subset)))
    return _rows_to_dicts(sample)


def intent_distribution(df: pd.DataFrame, category: str | None = None) -> dict[str, int]:
    """Return a mapping of intent to row count, optionally scoped to a category."""
    subset = _filter(df, category=category)
    counts = subset["intent"].value_counts()
    return {str(intent): int(count) for intent, count in counts.items()}


def search_examples(df: pd.DataFrame, query: str, n: int = 5) -> list[dict[str, str]]:
    """Return up to ``n`` rows whose customer instruction contains ``query``."""
    matches = _filter(df, text_contains=query)
    return _rows_to_dicts(matches.head(n))
