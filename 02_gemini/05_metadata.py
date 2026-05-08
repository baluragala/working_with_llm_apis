"""
02_gemini/05_metadata.py
-----------------------------------------------------------------
The response object carries more than `.text`. In production you almost always
care about:

    candidates       - all candidate completions (you usually get 1)
    finish_reason    - why the model stopped (STOP, MAX_TOKENS, SAFETY, ...)
    usage_metadata   - prompt / candidate / total token counts (for $$$ tracking)
    safety_ratings   - per-category safety classifications (when relevant)

Always log usage_metadata. It is the only honest answer to "how much did this cost?"
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Give me three reasons to learn Python.",
    config=types.GenerateContentConfig(
        max_output_tokens=200,
        candidate_count=1,        # try 2 if the model variant supports it
    ),
)

print("=== Text ===")
print(response.text)
print()

print("=== Candidates ===")
for i, cand in enumerate(response.candidates):
    print(f"  Candidate {i}: finish_reason = {cand.finish_reason}")
    if cand.safety_ratings:
        for sr in cand.safety_ratings:
            print(f"      safety: {sr.category.name:35} -> {sr.probability.name}")

print()
print("=== Token usage ===")
u = response.usage_metadata
print(f"  prompt_token_count     : {u.prompt_token_count}")
print(f"  candidates_token_count : {u.candidates_token_count}")
print(f"  total_token_count      : {u.total_token_count}")
