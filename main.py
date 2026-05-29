"""Entry point: CLI wiring for the data-analyst agent.

Parses --session (and the related --user) and wires the agent graph with a
SqliteSaver checkpointer so the same --session resumes its conversation on a
later run.
"""

import argparse
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph
from cli.repl import run_repl

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


def main() -> None:
    """Open the SQLite checkpointer, build the graph, and start the REPL."""
    args = _parse_args()
    with SqliteSaver.from_conn_string(str(_CHECKPOINT_DB)) as saver:
        graph = build_graph(checkpointer=saver)
        run_repl(graph=graph, session_id=args.session)


if __name__ == "__main__":
    main()
