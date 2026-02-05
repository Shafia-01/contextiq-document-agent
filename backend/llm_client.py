import os
from typing import Literal, Optional
from google import genai
from groq import Groq


def _create_gemini_client() -> genai.Client:
    """
    Construct a Google Gemini client using the modern ``google-genai`` SDK.

    - API key is read strictly from ``GEMINI_API_KEY``.
    - No keys are logged or printed.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Set it to a valid Gemini API key to use the Gemini provider."
        )
    return genai.Client(api_key=api_key)


def _create_groq_client() -> Groq:
    """
    Construct a Groq client.

    - API key is read strictly from ``GROQ_API_KEY``.
    - No keys are logged or printed.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Set it to a valid Groq API key to use the Groq provider."
        )
    return Groq(api_key=api_key)


def get_llm_client(provider: Optional[Literal["gemini", "groq"]] = None):
    """
    Return an LLM client for the requested provider.

    - **Default**: Gemini (per assessment requirements), resolved via:
        1. ``provider`` argument if passed, otherwise
        2. ``LLM_PROVIDER`` environment variable if set, otherwise
        3. The hard-coded default ``"gemini"``.
    - Supported providers: ``"gemini"``, ``"groq"``.

    This keeps the rest of the codebase provider-agnostic: callers only depend
    on this function and on the common LLM interface in ``qa.py``.
    """
    resolved = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if resolved == "gemini":
        return _create_gemini_client()
    if resolved == "groq":
        return _create_groq_client()
    raise ValueError(
        f"Unsupported LLM provider '{resolved}'. "
        "Expected one of: 'gemini', 'groq'."
    )
