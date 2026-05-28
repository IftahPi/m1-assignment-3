"""
Central configuration for Nebius Token Factory access.

The agent uses two models (dual strategy):
- a small, cheap model for the query *router* (classification only), and
- a larger model for the *agent* (tool-calling, reasoning, summarization).

Nebius exposes an OpenAI-compatible API, so we drive it through
``langchain_openai.ChatOpenAI`` with a custom ``base_url``.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

NEBIUS_BASE_URL: str = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")

# Dual-model strategy (override via env if a different Nebius id is desired).
# Router: small/cheap MoE (~3B active params) for fast query classification.
# Generator: larger model for tool-calling, reasoning, and summarization.
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct")


def get_api_key() -> str:
    """Return the Nebius API key, raising a clear error if it is not set."""
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at Nebius Token Factory.

    Args:
        model: The Nebius model id (e.g. ROUTER_MODEL or GENERATOR_MODEL).
        temperature: Sampling temperature; defaults to 0.0 for deterministic
            routing and reproducible analysis.

    Returns:
        A configured ChatOpenAI instance talking to the Nebius endpoint.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=NEBIUS_BASE_URL,
        api_key=get_api_key(),
    )
