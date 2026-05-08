# `03_provider_selection.py` — Concept Guide

## WHY (purpose)

Two failure modes haunt real LLM workloads:

1. **Vendor lock-in.** Your code couples to one company's quirks; when
   their pricing changes or their SLA breaks, switching is a project,
   not a config change.
2. **Vendor incidents.** A provider is down for an hour and your product
   is too.

Hugging Face's "Inference Providers" layer addresses both. The same model
identifier is hosted by several backends (Together, Fireworks, Replicate,
Cerebras, Hyperbolic, etc.); you can route by latency, cost, or
availability. The application code is unchanged across providers.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Inference provider** | A compute backend that hosts and serves models on Hugging Face's behalf. Each has its own pricing, latency profile, and feature support. |
| **`model_info(..., expand=["inferenceProviderMapping"])`** | Programmatic discovery: ask the Hub which providers currently serve a given model. Returns a dict keyed by provider name. |
| **`provider="auto"`** | Let HF pick. Convenient default for prototypes. |
| **`provider="<name>"`** | Pin to one (e.g. `"together"`, `"fireworks-ai"`, `"replicate"`). Use when you've benchmarked and want to control the choice. |
| **Hosted closed models on HF** | Some closed models (Anthropic's, OpenAI's, etc.) are accessible through HF's gateway — handy if you want a single token + single client across providers. Not free; pricing follows the upstream. |
| **Graceful failure per provider** | Provider availability varies by account, region, and model. Wrap each call in try/except so one bad provider doesn't take the whole script down. |

## HOW (code walkthrough)

```python
from huggingface_hub import InferenceClient, model_info
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

info = model_info(MODEL, expand=["inferenceProviderMapping"], token=token)
providers = list((info.inference_provider_mapping or {}).keys())
for p in providers:
    print(f"  - {p}")
```

The `expand=["inferenceProviderMapping"]` flag asks the Hub to include
which providers host this model. The list might be `["hf-inference",
"together", "fireworks-ai", "replicate", "cerebras", ...]` depending on
the day. Use `model_info` once at startup to drive a UI dropdown or a
config check.

```python
candidates = ["auto"] + providers[:2]
prompt = "In one sentence, why is Python popular?"

for provider in candidates:
    try:
        client = InferenceClient(token=token, provider=provider)
        resp = client.chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        print(f"[{provider}] {resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"[{provider}] FAIL: {type(e).__name__}: {e}")
```

A new `InferenceClient` per provider — only the `provider=` kwarg changes.
Everything else (model, messages, params) stays identical. That's the
no-rewrite-on-switch property in one screenshot.

**Run it:**

```bash
python 03_huggingface/03_provider_selection.py
```

You'll see the same answer rendered three ways — sometimes faster, sometimes
shorter, sometimes phrased differently. Pin to one provider in production
once you've measured; use `"auto"` while prototyping.
