"""
One-off connectivity check for Nebius Token Factory.

Run once after creating .env:  python smoke_test_nebius.py

It (1) lists candidate Llama model ids the account can see, then
(2) sends a tiny prompt to the configured router and generator models.
This is a manual boundary check, not part of the test suite.
"""

from openai import OpenAI

from nebius_client import (
    GENERATOR_MODEL,
    NEBIUS_BASE_URL,
    ROUTER_MODEL,
    get_api_key,
    make_llm,
)


def list_candidate_models() -> None:
    """Print Llama models the account can access, to confirm exact ids."""
    client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=get_api_key())
    ids = sorted(m.id for m in client.models.list().data)
    print(f"Total models visible: {len(ids)}")
    for model_id in ids:
        if "llama" in model_id.lower():
            print("  ", model_id)


def ping(model: str) -> None:
    """Send a one-token-ish prompt and print the reply."""
    reply = make_llm(model).invoke("Reply with exactly the word: OK")
    print(f"[{model}] -> {reply.content!r}")


if __name__ == "__main__":
    print("=== Llama models available on this account ===")
    list_candidate_models()
    print("\n=== Router model ===")
    ping(ROUTER_MODEL)
    print("=== Generator model ===")
    ping(GENERATOR_MODEL)
    print("\nSmoke test complete.")
