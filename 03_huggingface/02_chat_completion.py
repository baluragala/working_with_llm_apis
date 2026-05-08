"""
03_huggingface/02_chat_completion.py
-----------------------------------------------------------------
Recreate the Gemini "lighthouse keeper" example using HF InferenceClient.

Key contrasts with Gemini:
    - Roles: "system" / "user" / "assistant"  (Gemini uses "user" / "model")
    - System prompt is a regular message with role="system", not a separate field.
    - Generation parameters live as top-level kwargs, not inside a config object.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
client = InferenceClient(token=os.environ["HF_TOKEN"])

PROMPT = "Write one opening line for a short story about a lighthouse keeper."
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ---- Deterministic ----------------------------------------------------
boring = client.chat_completion(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.0,
    max_tokens=60,
)
print("=== temperature=0.0 ===")
print(boring.choices[0].message.content)
print()

# ---- Creative + system role ------------------------------------------
creative = client.chat_completion(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a hard-boiled noir detective novelist. "
                "Every sentence is short, dark, and cynical."
            ),
        },
        {"role": "user", "content": PROMPT},
    ],
    temperature=1.2,
    top_p=0.95,
    max_tokens=60,
)
print("=== noir detective, temperature=1.2 ===")
print(creative.choices[0].message.content)
