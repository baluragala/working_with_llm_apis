# `02_parameters.py` — Concept Guide

## WHY (purpose)

The default parameters are tuned for "looks reasonable on average" — fine
for a demo, wrong for a product. Three concrete needs almost always come
up in real work:

* **Reproducibility / determinism.** If a downstream test compares model
  output to a fixed string, you need temperature 0.
* **Variety.** A creative-writing tool wants different output every time;
  a low temperature makes it sound robotic.
* **Persona / style enforcement.** "Always answer as a helpful assistant"
  belongs in the system instruction, not stuffed into every user prompt.

This script shows the same prompt rendered three ways so the effect of
each knob is visible side by side.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Temperature** | Scales the logits before softmax. `0.0` makes the highest-probability token always win (deterministic). Higher values flatten the distribution → more surprising completions. Range 0.0–2.0. |
| **`top_p` (nucleus sampling)** | Keeps the smallest set of top tokens whose cumulative probability ≥ `top_p`, then samples within that set. `top_p=0.95` discards the unlikely long tail. Range 0.0–1.0. |
| **Why not tune both** | Temperature and top_p both restrict randomness, in different ways. Tuning both at once interacts in confusing ways. Pick one. Most production setups touch only temperature. |
| **`max_output_tokens`** | Hard ceiling on the model's response length. The model can stop earlier (`finish_reason=STOP`) or be cut off here (`finish_reason=MAX_TOKENS`). |
| **`system_instruction`** | A persistent role/style the model adopts. Conceptually separate from the user's prompt. In Gemini it's a config field, not a `role: "system"` message (that's OpenAI/HF style). |
| **`GenerateContentConfig`** | The container object where all of these live. Pass it via `config=` on `generate_content`. |

## HOW (code walkthrough)

```python
from google.genai import types
config = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=60,
)
boring = client.models.generate_content(
    model="gemini-2.5-flash", contents=PROMPT, config=config,
)
```

Run this twice — the output is identical. Temperature 0 is a strong
guarantee of reproducibility for the same prompt and model version.

```python
config = types.GenerateContentConfig(
    temperature=1.5, top_p=0.95, max_output_tokens=60,
)
```

High temperature widens the next-token distribution; `top_p` then chops
off the long tail of nonsense tokens. The model stays creative without
going off the rails. Run twice — output differs each time.

```python
config = types.GenerateContentConfig(
    system_instruction=(
        "You are a hard-boiled noir detective novelist. "
        "Every sentence is short, dark, and cynical."
    ),
    temperature=0.9, max_output_tokens=60,
)
```

The system instruction reshapes voice and style without bloating the user
prompt. The same `PROMPT` string ("write one opening line about a
lighthouse keeper") now produces a noir line instead of a generic one.

**Run it:**

```bash
python 02_gemini/02_parameters.py
```

Read the three blocks side by side. Then change `temperature` to `0.0` in
block 2 and rerun — the creative output collapses back to the boring one.
That's the whole intuition.
