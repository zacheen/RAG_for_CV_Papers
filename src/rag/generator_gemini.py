"""Generate answers using Google AI Studio Gemini with retrieved context.

This is a free-tier alternative to the Ollama-based generator. It uses the
Google AI Studio (Gemini API) free quota so the project can run without a GCP
VM hosting Ollama.

Set the GOOGLE_API_KEY environment variable to your Google AI Studio API key
before using this module. Get one for free at https://aistudio.google.com/apikey
"""

import os

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL, RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


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
