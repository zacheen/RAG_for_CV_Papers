"""Generate answers using Google AI Studio Gemini with retrieved context.

This is a free-tier alternative to the Ollama-based generator. It uses the
Google AI Studio (Gemini API) free quota so the project can run without a GCP
VM hosting Ollama.

Set the GOOGLE_API_KEY environment variable to your Google AI Studio API key
before using this module. Get one for free at https://aistudio.google.com/apikey
"""

import datetime
import os

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL, RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from src.rag.tools import get_tools, time_range_state


PRE_RAG_SYSTEM_PROMPT = (
    "You are an intent extractor for a computer-vision paper RAG chatbot. "
    "Inspect the user's latest message and call the provided tools when appropriate:\n"
    "- set_time_range: when the user wants to filter papers by publication date.\n"
    "- clear_time_range: when the user wants to remove an existing date filter.\n"
    "- download_cited_papers: ONLY when the user explicitly asks to download "
    "the references of a specific paper by arXiv id. Do NOT call this for "
    "general questions about papers; the application handles inline citation "
    "downloads automatically.\n"
    "If no tool applies, do nothing and return a short acknowledgement. "
    "Today's date is {today}. Convert relative phrases (\"last week\", "
    "\"March 2025\", \"since 2024\") to absolute ISO dates using today as the "
    "reference. The current active time range is: {current_range}."
)


def _get_client() -> genai.Client:
    """Create a Gemini client using the GOOGLE_API_KEY env var."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "Get a free key at https://aistudio.google.com/apikey "
            "and export it before running."
        )
    return genai.Client(api_key=api_key)


def _build_contents(question: str, context: str,
                    chat_history: list[dict] | None) -> list[types.Content]:
    """Convert OpenAI-style chat messages into Gemini Content objects."""
    contents: list[types.Content] = []

    if chat_history:
        for message in chat_history:
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=message["content"])],
                )
            )

    user_content = RAG_USER_TEMPLATE.format(context=context, question=question)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_content)],
        )
    )
    return contents


def generate_answer(question: str, context: str,
                    model: str = GEMINI_MODEL,
                    chat_history: list[dict] | None = None) -> str:
    """Generate an answer using Gemini with RAG context.

    Args:
        question: User question.
        context: Formatted context string from retrieved chunks.
        model: Gemini model name (e.g. "gemini-1.5-flash").
        chat_history: Optional prior conversation messages.

    Returns:
        Generated answer string.
    """
    client = _get_client()
    contents = _build_contents(question, context, chat_history)

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=RAG_SYSTEM_PROMPT),
    )
    return response.text or ""


def generate_answer_stream(question: str, context: str,
                           model: str = GEMINI_MODEL,
                           chat_history: list[dict] | None = None):
    """Stream an answer using Gemini with RAG context.

    Args:
        question: User question.
        context: Formatted context from retrieved chunks.
        model: Gemini model name.
        chat_history: Optional prior conversation messages.

    Yields:
        Token strings as they are generated.
    """
    client = _get_client()
    contents = _build_contents(question, context, chat_history)

    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=RAG_SYSTEM_PROMPT),
    )
    for chunk in stream:
        token = chunk.text
        if token:
            yield token


def run_pre_rag_pass(prompt: str, model: str = GEMINI_MODEL) -> None:
    """Stage 1 of the function-calling flow.

    Sends the user prompt to Gemini with the three RAG tools attached and
    lets Automatic Function Calling fire any that the model picks. Tools
    mutate module-level state in :mod:`src.rag.tools` and
    :mod:`src.rag.download_state`; this function ignores the model's text
    output.

    Errors (missing API key, network failure, model error) are swallowed and
    logged via the return value being a no-op so the main RAG flow can
    proceed even when the pre-pass fails. Streamlit-side code can decide
    whether to surface a warning.

    Args:
        prompt: The user's latest chat message.
        model: Gemini model name.
    """
    if not prompt or not prompt.strip():
        return

    try:
        client = _get_client()
    except RuntimeError:
        # No API key — caller already shows a friendly message in the main path.
        return

    today = datetime.date.today().isoformat()
    current = time_range_state.to_dict()
    if current["start_date"] and current["end_date"]:
        current_range = f"{current['start_date']} -> {current['end_date']}"
    else:
        current_range = "All time (no filter)"

    system_instruction = PRE_RAG_SYSTEM_PROMPT.format(
        today=today, current_range=current_range
    )

    try:
        client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=get_tools(),
            ),
        )
    except Exception:
        # AFC / network errors should not break the main RAG turn.
        return
