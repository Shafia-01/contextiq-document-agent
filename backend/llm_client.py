import os
from google import genai
from groq import Groq


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)
    return client


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return Groq(api_key=api_key)


def get_llm_client(provider: str = "gemini"):
    """
    Returns the configured LLM client.
    Gemini is the default provider as per assessment requirements.
    """
    if provider == "gemini":
        return get_gemini_client()
    elif provider == "groq":
        return get_groq_client()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
