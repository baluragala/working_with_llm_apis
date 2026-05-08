# `test_connectivity.py` — Concept Guide

## WHY (purpose)

Before any LLM workshop or production deployment, you need to be sure that:

1. The API keys you generated are actually loaded by your program.
2. Each provider can be reached over the network from your machine.
3. The keys have permission to call the models you intend to use.

Most of the "my code doesn't work" hours that beginners lose are spent on
some combination of: a typo in the variable name, the `.env` file in the
wrong folder, a key without billing enabled, or a corporate proxy blocking
the call. A standalone connectivity probe surfaces all four classes of
failure cheaply, in one screen, before you go further.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Environment variables** | Per-process key=value pairs read from `os.environ`. The standard place to keep secrets that should *not* live in source code. |
| **`.env` file + `python-dotenv`** | A local convention: a file named `.env` at the project root holds `KEY=value` lines. `load_dotenv()` reads it and copies the values into `os.environ` for the current process. The file is added to `.gitignore` so secrets never reach version control. |
| **Provider SDK clients** | Each vendor ships a thin Python wrapper: `google.genai.Client`, `huggingface_hub.InferenceClient`, `openai.OpenAI`. They take the key in their constructor and expose a `generate_content` / `chat_completion` / `chat.completions.create` method. |
| **Smoke test** | The smallest possible request that proves the round-trip works. Asking the model to "reply with the word pong" exercises authentication, network, model availability, and response parsing in one shot. |
| **Hard-coded keys (anti-pattern)** | Putting the literal key string in source code. Bots scan public repos for these patterns within minutes; a single push can leak a billing-attached key. |

## HOW (code walkthrough)

```python
from dotenv import load_dotenv
loaded = load_dotenv()           # reads .env  → os.environ
```

A single call mutates the process environment. Returns `True` if a `.env`
was found — useful as a sanity print so you know your file is in the right
place.

```python
key = os.getenv("GEMINI_API_KEY")
if not key or key.startswith("your_"):
    print("Gemini : SKIP (no GEMINI_API_KEY set)")
    return
```

`os.getenv` returns `None` when the variable is missing, instead of raising.
The `startswith("your_")` check catches the common case of forgetting to
replace the placeholder from `.env.example`.

```python
from google import genai
client = genai.Client(api_key=key)
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly the word: pong",
)
print(f"Gemini : OK -> {resp.text.strip()!r}")
```

The same pattern repeats for Hugging Face (`InferenceClient(token=...)` →
`chat_completion(...)`) and OpenAI (`OpenAI(api_key=...)` →
`chat.completions.create(...)`). The shape is intentionally the same: a
client built with a credential, one method to send a tiny prompt, one
attribute to read the reply.

```python
# OPENAI_API_KEY = "sk-proj-abc123..."   ← never do this
```

The commented-out anti-pattern is part of the lesson. Keys committed even
once must be rotated; deleting the line later does not erase it from git
history.

**Run it:**

```bash
cp 01_secure_setup/.env.example .env   # then paste real keys into .env
python 01_secure_setup/test_connectivity.py
```

You'll see a `OK / FAIL / SKIP` line per provider plus the model's reply.
Every other script in the repo assumes this one passes first.
