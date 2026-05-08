"""
02_gemini/02_parameters.py
-----------------------------------------------------------------
Same prompt, different generation parameters. Run this twice with the same
seed and you'll see the boring story is identical, but the creative one varies.

Parameters demonstrated:
    temperature      0.0 - 2.0   How "random" the next token is. 0 = deterministic.
    top_p            0.0 - 1.0   Nucleus sampling cutoff. Keep tokens until
                                 cumulative probability reaches top_p.
    max_output_tokens          Hard ceiling on response length.
    system_instruction         Persistent role/style (separate from the user prompt).

Rule of thumb: tune ONE of temperature / top_p, not both.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = "Write one opening line for a short story about a lighthouse keeper."

# ---- 1. Deterministic (temperature 0) -------------------------------
boring = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=60,
    ),
)
print("=== temperature=0.0 (deterministic) ===")
print(boring.text)
print()

# ---- 2. High temperature, creative ---------------------------------
creative = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        temperature=1.5,
        top_p=0.95,
        max_output_tokens=60,
    ),
)
print("=== temperature=1.5, top_p=0.95 (creative) ===")
print(creative.text)
print()

# ---- 3. With a system instruction ----------------------------------
# system_instruction sets the model's persona / rules. It is sent on every
# turn but is conceptually separate from the user's contents.
styled = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=PROMPT,
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are a hard-boiled noir detective novelist. "
            "Every sentence is short, dark, and cynical."
        ),
        temperature=0.9,
        max_output_tokens=60,
    ),
)
print("=== system_instruction = noir detective ===")
print(styled.text)
