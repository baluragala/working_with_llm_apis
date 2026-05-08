# `02_compare_latency.py` — Concept Guide

## WHY (purpose)

The "hosted vs local" debate stops being abstract once you watch the same
prompt run on three backends with a stopwatch. The numbers will surprise
some learners — a hosted Llama-3.1-8B on a fast provider may *beat* a
local 1.1B model on a laptop's CPU; a small local model on Apple Silicon
may beat a slow free-tier hosted call.

This script is a teaching probe, not a benchmark. The goal is to show
**the shape of the trade-off**, not to publish a leaderboard.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Wall-clock latency** | Time from "I called the function" to "I have the answer". Includes network, queueing, the model's actual computation, and SDK overhead. The only number that matters to a user. |
| **Cold start vs steady state** | The first call after a process starts (or after a model has been off provider's hot pool) is slow because of model loading. Later calls are fast. Numbers from the *first* call mislead. |
| **First-token latency vs total latency** | A streaming UI cares about time-to-first-token. A batch job cares about total time. This script measures total only — adapt for your case. |
| **Why we wrap each backend in a function** | Lazy imports: only the providers you actually call get imported. Lets the script run even if one SDK isn't installed (e.g. you can still time Gemini if `transformers` is missing). |
| **`time.time()`** | Wall-clock timing in seconds. Good enough for human-visible latencies; for sub-millisecond stuff use `time.perf_counter()`. |
| **Try/except per backend** | One backend failing (no token, model gated, network down) shouldn't tank the whole comparison. Catch and report; move on. |

## HOW (code walkthrough)

```python
PROMPT = "Explain prompt engineering in two sentences."

def time_call(label, fn):
    t0 = time.time()
    try:
        out = fn()
        dt = time.time() - t0
        print(f"\n[{label}] {dt:.2f}s")
        print(out.strip()[:300])
    except Exception as e:
        print(f"\n[{label}] FAILED: {type(e).__name__}: {e}")
```

A single timing harness used three times. The reason it lives in a tiny
helper is so the call sites stay short and identical-looking — easier to
read on a slide.

```python
def gemini_call():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(model="gemini-2.5-flash", contents=PROMPT)
    return r.text

def hf_call():
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    r = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200,
    )
    return r.choices[0].message.content

def local_call():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    # ... pick device, load TinyLlama, apply_chat_template, generate ...
```

Three closures, one each per backend. Imports are inside the function so
a missing dependency only kills its own backend.

```python
time_call("Gemini  ", gemini_call)
time_call("HF      ", hf_call)
time_call("Local TL", local_call)
```

Run all three. Read the numbers. Don't quote them as benchmarks — re-run
five times and you'll get a different ordering. The point is the
**trade-off shape**: hosted is fast and abstracts away hardware; local is
private and predictable but bound by your machine.

**Run it:**

```bash
python 04_local_model/02_compare_latency.py
```

Run it twice. The second run will be faster on every backend (warm cache,
warm model) — that gap between cold and warm runs *is* one of the lessons.
