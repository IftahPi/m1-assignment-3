"""Per-user semantic memory: a freeform Markdown context file.

Each user gets a single ``profiles/<user_id>.md`` blob distilled from prior
sessions — names, recurring interests, stated preferences. Distinct from the
episodic conversation log (which the SQLite checkpointer keeps).

Reads happen at the start of a personal turn (loaded into the agent state),
writes happen at the end of a session via :func:`summarize_session`, which
asks the generator LLM to update the profile from the new transcript.
"""

from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

_PROFILES_DIR: Path = Path(__file__).resolve().parent.parent / "profiles"


SUMMARY_SYSTEM_PROMPT: str = """\
You distill a user's persistent profile from a single chat session.

You receive: (1) the EXISTING user profile (may be empty), and (2) the NEW \
conversation transcript. Produce an UPDATED profile in plain Markdown that \
captures DISTILLED FACTS about the user — name, recurring interests, stated \
preferences, important context they have shared. It is NOT a transcript and \
NOT a recap of questions asked; it is what is TRUE about the user.

Rules:
- Keep existing facts unless contradicted by new information.
- Add only new, durable facts. Skip one-off questions that don't reveal \
anything about the user.
- Use short sentences or bullets under topical headers ("Name", "Interests", \
"Preferences", "Notes").
- If there is nothing worth saving, return the existing profile unchanged.
- Output plain Markdown only — no tables, no asterisks for bold.
"""


def _profile_path(user_id: str) -> Path:
    """Return the on-disk path of a given user's profile file."""
    return _PROFILES_DIR / f"{user_id}.md"


def load_profile(user_id: str) -> str:
    """Return the user's profile Markdown, or an empty string if no profile exists."""
    path = _profile_path(user_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_profile(user_id: str, text: str) -> None:
    """Write the user's profile Markdown to disk, creating the directory if needed."""
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    _profile_path(user_id).write_text(text, encoding="utf-8")


def _format_transcript(messages: list[AnyMessage]) -> str:
    """Render the session's human and final-AI replies for the summary prompt."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            text = str(message.content).strip()
            if text:
                lines.append(f"user: {text}")
        elif isinstance(message, AIMessage) and not message.tool_calls:
            text = str(message.content).strip()
            if text:
                lines.append(f"assistant: {text}")
    return "\n".join(lines) or "(empty session)"


def summarize_session(
    messages: list[AnyMessage],
    prior_profile: str,
    llm: object,
) -> str:
    """Distill an updated profile from the session messages + the prior profile.

    Args:
        messages: The session's messages (Human + AI replies). Tool messages
            and tool-call AIMessages are skipped for brevity.
        prior_profile: The existing profile Markdown (may be empty).
        llm: An LLM with ``.invoke(messages) -> AIMessage``. Injected for
            testability; in production it's a no-tools ChatOpenAI for the
            generator model.

    Returns:
        The updated profile Markdown.
    """
    user_content = (
        f"EXISTING PROFILE:\n{prior_profile or '(none)'}\n\n"
        f"NEW SESSION TRANSCRIPT:\n{_format_transcript(messages)}\n\n"
        "Return the updated profile in plain Markdown."
    )
    response = llm.invoke(
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)
