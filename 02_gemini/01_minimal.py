"""
02_gemini/01_minimal.py
-----------------------------------------------------------------
The smallest possible Gemini request. Two required arguments:
    - model    : which Gemini model to use
    - contents : what you're asking it

Everything else (temperature, system instructions, tools, etc.) is optional
and layered on top.

Docs: https://ai.google.dev/gemini-api/docs/quickstart
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain prompt engineering in two sentences.",
)

print("Response text:")
print(response.text)
