# `01_inference_client.py` — Concept Guide

## WHY (purpose)

There are dozens of inference providers — Hugging Face, Together AI,
Fireworks, Replicate, Cerebras, Hyperbolic, the model owners themselves.
Each ships its own SDK with its own conventions. Building against one
locks you in; building against several is busywork.

`InferenceClient` is Hugging Face's answer: one Python class, many
backends. The same code that runs against HF's own inference can be
flipped to Together or Fireworks by changing a single argument. That makes
it the natural starting point for the "open / closed / hosted / wherever"
section of the workshop.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Hugging Face Hub** | A registry of models (and datasets and Spaces). Models are addressed as `org/model-name` — the same string used in URLs like `huggingface.co/meta-llama/Llama-3.1-8B-Instruct`. |
| **Inference Providers** | A routing layer Hugging Face runs in front of multiple compute providers. You pick a model; HF (or you) picks the provider that serves it. |
| **`InferenceClient`** | The Python entry point. One client, many tasks: chat, embeddings, summarisation, image classification, ASR, etc. |
| **`chat_completion`** | The chat method. Takes a `messages` list with `role` ∈ {`system`, `user`, `assistant`}. The shape is intentionally OpenAI-compatible — code written for OpenAI mostly drops in. |
| **HF token (`HF_TOKEN`)** | Your auth credential. A "read"-scope token is enough for inference; "write" is only needed if you push models. Some models are *gated* (you must accept terms on the model page) before any token works for them. |
| **`response.choices[0].message.content`** | The reply text. Same path as OpenAI. `finish_reason` and `usage` live on the same response object. |

## HOW (code walkthrough)

```python
from huggingface_hub import InferenceClient
client = InferenceClient(token=os.environ["HF_TOKEN"])
```

A single client handle. No provider specified yet — HF will route it for
you. The `token` is mandatory for non-public usage; for ungated public
models it's optional but recommended (avoids stricter rate limits on
anonymous requests).

```python
response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Explain prompt engineering in two sentences."},
    ],
    max_tokens=200,
)
```

Notice the difference from Gemini's API: the model is a single string
identifier (no separate "client per model"), and the prompt is a
`messages` list with explicit roles instead of a free-form `contents`.
This is the OpenAI shape; you'll see it again in `05_openai/`.

```python
print(response.choices[0].message.content)
print("Finish reason :", response.choices[0].finish_reason)
print("Token usage   :", response.usage)
```

The same metadata fields (finish reason, usage) you saw in Gemini's
`response.usage_metadata` — different attribute names, same idea: log
them.

**Run it:**

```bash
python 03_huggingface/01_inference_client.py
```

Then change the model to `mistralai/Mistral-7B-Instruct-v0.3` and rerun.
No other line changes — that is the InferenceClient value proposition.
