"""Entry point: CLI wiring for the data-analyst agent.

Parses --session (episodic memory via SqliteSaver) and --user (semantic
memory via a per-user Markdown profile). When the REPL exits, the session's
messages are distilled into an updated profile via :func:`summarize_session`.
"""

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph
from agent.profile import load_profile, save_profile, summarize_session
from cli.repl import run_repl
from nebius_client import GENERATOR_MODEL, make_llm

_CHECKPOINT_DB: Path = Path(__file__).resolve().parent / "checkpoints.sqlite"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--user`` defaults to whatever ``--session`` is."""
    parser = argparse.ArgumentParser(
        description="Data-analyst agent for the Bitext Customer Service dataset."
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session id; the same id resumes the prior conversation (default: 'default').",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User id for the persistent profile; defaults to --session.",
    )
    args = parser.parse_args()
    if args.user is None:
        args.user = args.session
    return args


def _distill_profile_on_exit(graph: object, session_id: str, user_id: str) -> None:
    """After the REPL ends, update the user's semantic profile from the session.

    Pulls the final state from the checkpointer, runs the summary LLM against
    the prior profile + this session's messages, and writes the updated
    profile to ``profiles/<user_id>.md``. Silent if the session had no human
    turns (nothing new to distill).
    """
    state = graph.get_state({"configurable": {"thread_id": session_id}})
    messages = state.values.get("messages", [])
    if not any(isinstance(m, HumanMessage) for m in messages):
        return

    prior_profile = load_profile(user_id)
    summary_llm = make_llm(GENERATOR_MODEL, temperature=0.2)  # plain text, no tools
    updated_profile = summarize_session(messages, prior_profile, summary_llm)
    save_profile(user_id, updated_profile)
    print(f"\n💾 Updated profile for user '{user_id}' saved to profiles/{user_id}.md")


def main() -> None:
    """Open the SQLite checkpointer, build the graph, run the REPL, then distill."""
    args = _parse_args()
    with SqliteSaver.from_conn_string(str(_CHECKPOINT_DB)) as saver:
        graph = build_graph(checkpointer=saver)
        run_repl(graph=graph, session_id=args.session, user_id=args.user)
        _distill_profile_on_exit(graph, args.session, args.user)


if __name__ == "__main__":
    main()
