"""
01_secure_setup/test_connectivity.py
-----------------------------------------------------------------
Goal: Prove all three API keys are loaded and reachable, and demonstrate
the right way to handle keys (env vars, never hardcoded).

Run:
    python 01_secure_setup/test_connectivity.py

What you'll see:
    A tick or cross for each provider, plus a one-line response from the model
    so you know the round trip actually worked - not just authentication.
"""

import os
import sys
from dotenv import load_dotenv

# Load variables from .env into os.environ. Returns True if a .env file was found.
loaded = load_dotenv()
print(f".env loaded: {loaded}\n")


def check_gemini() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key or key.startswith("your_"):
        print("Gemini       : SKIP (no GEMINI_API_KEY set)")
        return
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly the word: pong",
        )
        print(f"Gemini       : OK   -> {resp.text.strip()!r}")
    except Exception as e:
        print(f"Gemini       : FAIL -> {type(e).__name__}: {e}")


def check_huggingface() -> None:
    token = os.getenv("HF_TOKEN")
    if not token or token.startswith("your_"):
        print("HuggingFace  : SKIP (no HF_TOKEN set)")
        return
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        # Use a small, widely-available chat model on the default provider.
        resp = client.chat_completion(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
            max_tokens=10,
        )
        print(f"HuggingFace  : OK   -> {resp.choices[0].message.content.strip()!r}")
    except Exception as e:
        print(f"HuggingFace  : FAIL -> {type(e).__name__}: {e}")


def check_openai() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key or key.startswith("your_"):
        print("OpenAI       : SKIP (no OPENAI_API_KEY set)")
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
            max_tokens=10,
        )
        print(f"OpenAI       : OK   -> {resp.choices[0].message.content.strip()!r}")
    except Exception as e:
        print(f"OpenAI       : FAIL -> {type(e).__name__}: {e}")


# -------- Anti-pattern (do NOT do this) ------------------------------
# OPENAI_API_KEY = "sk-proj-abc123...."     # Hardcoded in source
# client = OpenAI(api_key=OPENAI_API_KEY)   # One commit, one git push,
#                                           # and your key is on the public
#                                           # internet forever. Bots scrape
#                                           # GitHub for these within minutes.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing API connectivity. SKIP means no key is configured.\n")
    check_gemini()
    check_huggingface()
    check_openai()
    print("\nDone. If anything failed, check the error and your .env file.")
