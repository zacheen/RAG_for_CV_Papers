"""Generate answers using Ollama with retrieved context."""

import ollama

from src.config import OLLAMA_MODEL, RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


def generate_answer(question: str, context: str,
                    model: str = OLLAMA_MODEL,
                    chat_history: list[dict] | None = None) -> str:
    """Generate an answer using the LLM with RAG context.

    Args:
        question: User question.
        context: Formatted context string from retrieved chunks.
        model: Ollama model name.
        chat_history: Optional prior conversation messages.

    Returns:
        Generated answer string.
    """
    user_content = RAG_USER_TEMPLATE.format(context=context, question=question)

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_content})

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def generate_answer_stream(question: str, context: str,
                           model: str = OLLAMA_MODEL,
                           chat_history: list[dict] | None = None):
    """Stream an answer using the LLM with RAG context.

    Args:
        question: User question.
        context: Formatted context from retrieved chunks.
        model: Ollama model name.
        chat_history: Optional prior conversation messages.

    Yields:
        Token strings as they are generated.
    """
    user_content = RAG_USER_TEMPLATE.format(context=context, question=question)

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_content})

    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
