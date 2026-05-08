# `02_chat_completion.py` — Concept Guide

## WHY (purpose)

`02_gemini/02_parameters.py` showed temperature, top_p, and system
instructions on Gemini. Real teams use multiple providers, so the same
intuitions need to map across SDKs. This script recreates the Gemini
example on Hugging Face — same prompt, same parameters, same observable
outcome — so the SDK differences are the *only* thing in front of the
learner.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Roles in `messages`** | `"system"`, `"user"`, `"assistant"` — the OpenAI/HF triple. The system prompt is *just another message*, not a separate config field as it is in Gemini. |
| **Inline parameters** | HF's `chat_completion` takes generation parameters as plain kwargs (`temperature=`, `top_p=`, `max_tokens=`). No `GenerateContentConfig` wrapper. |
| **`max_tokens` vs `max_output_tokens`** | Same idea (cap on response length), different name across providers. Easy to mistype between SDKs — keep the docs handy. |
| **Determinism with `temperature=0`** | Same guarantee as in Gemini: deterministic decoding. Use it for any test you want to assert on a fixed string. |
| **Voice / persona via system role** | Putting "You are a noir novelist" in `role: "system"` is the OpenAI/HF analogue of Gemini's `system_instruction`. The result is the same — the model adopts that voice without bloating the user prompt. |

## HOW (code walkthrough)

```python
client = InferenceClient(token=os.environ["HF_TOKEN"])
PROMPT = "Write one opening line for a short story about a lighthouse keeper."
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
```

Same prompt and model held constant across all three blocks so only
parameters vary.

```python
boring = client.chat_completion(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    temperature=0.0,
    max_tokens=60,
)
print(boring.choices[0].message.content)
```

Deterministic block. Run twice — output is identical. Compare with
`02_gemini/02_parameters.py` block 1 — same idea, different SDK shape.

```python
creative = client.chat_completion(
    model=MODEL,
    messages=[
        {"role": "system",
         "content": "You are a hard-boiled noir detective novelist. "
                    "Every sentence is short, dark, and cynical."},
        {"role": "user", "content": PROMPT},
    ],
    temperature=1.2,
    top_p=0.95,
    max_tokens=60,
)
```

The system prompt is just an extra entry at the top of `messages`. The
generation parameters sit at the same level as `model` — no config
object. This is the only structural difference from Gemini worth
remembering.

**Run it:**

```bash
python 03_huggingface/02_chat_completion.py
```

Now open `02_gemini/02_parameters.py` side by side. The conceptual
controls (temperature, top_p, system role, max_tokens) are 1:1; only the
syntactic packaging differs. Internalising that mapping is the workshop's
"cross-platform mental model" goal.
