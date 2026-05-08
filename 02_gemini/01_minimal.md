# `01_minimal.py` — Concept Guide

## WHY (purpose)

Every API has a "hello world" — the smallest amount of code that proves
the SDK is installed, the key is valid, and the model is reachable. For
Gemini that's a single `generate_content` call with two arguments. Once a
learner has this working, every later feature (parameters, multimodal,
tools, structured output) is a small addition on top of the same skeleton.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **`google-genai` SDK** | The official Google GenAI Python client. `from google import genai` exposes `genai.Client`, the entry point. Successor to the older `google-generativeai` package. |
| **`Client`** | A configured handle that knows your API key and which Google endpoint to talk to. Construct it once per process and reuse. |
| **`client.models.generate_content`** | The synchronous one-shot generation call. It returns a fully populated response — there's no streaming, no chat history, nothing else to set up. |
| **`model` argument** | A string identifier like `"gemini-2.5-flash"` (fast, cheap) or `"gemini-2.5-pro"` (smarter, slower, costlier). Different models also support different features (e.g. some accept video, some don't). |
| **`contents` argument** | What you're asking the model. Accepts a plain string (this script), or a list mixing strings and `Part` objects (later scripts). |
| **`response.text`** | A convenience accessor that pulls the joined text out of `response.candidates[0].content.parts[*].text`. Equivalent to `response.choices[0].message.content` in OpenAI. |

## HOW (code walkthrough)

```python
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()                                          # .env  → os.environ
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
```

Two-line setup. `os.environ["..."]` (with brackets) raises immediately if
the key is missing — preferred over `.get()` here because there is no
sensible fallback for a missing API key.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain prompt engineering in two sentences.",
)
print(response.text)
```

The whole call is `model + contents`. No system prompt, no temperature, no
tools — those are all optional configuration that we layer on in later
scripts. `response.text` collapses the structured response into the single
string you usually want.

**Run it:**

```bash
python 02_gemini/01_minimal.py
```

If you see two sentences of explanation, the rest of the Gemini section
will work. If this fails, debug here before going on.
