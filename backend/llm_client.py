import os
import google.generativeai as genai
from groq import Groq

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    return genai

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("⚠️ GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)
