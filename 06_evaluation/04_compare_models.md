# `04_compare_models.py` — Concept Guide

## WHY (purpose)

Picking between two models on vibes is how teams ship the wrong one. The
discipline that beats vibes is *offline evaluation*: a fixed task, a
fixed reference, the same prompt sent to each candidate model, scored
with one or more metrics, with the raw outputs preserved so you can
re-score later without re-spending API budget.

This script is that pipeline at minimum size — small enough to read in
one screen, structurally identical to a serious eval harness.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Offline evaluation pipeline** | task → reference → generate from each model → persist outputs → score → review numbers and read texts. The order matters; persisting before scoring is what lets you iterate on metrics later. |
| **Reference (gold answer)** | The "right" output, written by a human. The whole evaluation is relative to this. Garbage references → garbage metrics. |
| **Holding the prompt constant** | Every model sees the same prompt. The only thing that varies is the model. Otherwise you're benchmarking prompts, not models. |
| **Persisting outputs** | Saving the raw outputs to disk (JSON here) makes re-scoring with a new metric a free operation. Without persistence, every metric tweak is a re-spend on API calls. |
| **`pathlib.Path(__file__).parent`** | A simple way to write outputs next to the script regardless of where you run it from. |
| **ROUGE F1 across models** | A side-by-side table of `rouge1-F`, `rouge2-F`, `rougeL-F` for each candidate. The model with the highest numbers is *probably* the best, but not always — see the warning below. |
| **Why "always read the outputs"** | Metrics correlate with quality but don't replace it. A model can game ROUGE by parroting reference vocabulary while being factually wrong. The eyeball pass is non-negotiable. |

## HOW (code walkthrough)

```python
ARTICLE   = "...Apollo program..."
PROMPT    = f"Summarize the following in one sentence:\n\n{ARTICLE}"
REFERENCE = "NASA's Apollo program (1961-1972) landed humans on the Moon..."
OUTPUTS_FILE = Path(__file__).parent / "outputs.json"
```

Task, prompt, gold answer, and the path for cached outputs — all defined
up front so the rest of the script is just a pipeline.

```python
def from_gemini():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(model="gemini-2.5-flash", contents=PROMPT)
    return r.text.strip()

def from_hf():
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    r = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200,
    )
    return r.choices[0].message.content.strip()
```

Two adapters. Adding a third candidate is a five-line function call.

```python
outputs = {}
for label, fn in [("gemini-2.5-flash", from_gemini), ("llama-3.1-8b-hf", from_hf)]:
    try:
        outputs[label] = fn()
    except Exception as e:
        outputs[label] = f"ERROR: {e}"

OUTPUTS_FILE.write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
```

Run all candidates, capture errors per-candidate, persist the raw
results. Re-running the script later with `outputs.json` already present
is a one-line change away (read instead of generate) — that's the
"avoid re-spending API budget" pattern.

```python
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
print(f"{'Model':<22} {'rouge1-F':>10} {'rouge2-F':>10} {'rougeL-F':>10}")
for label, text in outputs.items():
    s = scorer.score(REFERENCE, text)
    print(f"{label:<22} {s['rouge1'].fmeasure:>10.3f} {s['rouge2'].fmeasure:>10.3f} {s['rougeL'].fmeasure:>10.3f}")

print("\nReference:")
print("  " + REFERENCE)
print("\nGenerated:")
for label, text in outputs.items():
    print(f"  [{label}] {text[:200]}")
```

A scoring table followed by the actual texts. The texts are deliberately
printed *after* the scores — read both, then decide.

**Run it:**

```bash
python 06_evaluation/04_compare_models.py
```

Then open `outputs.json` and re-run only the scoring portion (you can
copy it into a one-liner notebook). That decoupling — *generate-once,
score-many* — scales to dozens of metrics and hundreds of test items
without changing shape. It is the bare-bones template real eval harnesses
elaborate on.
