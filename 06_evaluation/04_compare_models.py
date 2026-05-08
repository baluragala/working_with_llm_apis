"""
06_evaluation/04_compare_models.py
-----------------------------------------------------------------
End-to-end: generate summaries from two providers and score them against
a human reference. This is the basic shape of every offline eval pipeline.

Pipeline:
    1. Define a task (summarize this article).
    2. Define a reference (gold-standard answer).
    3. Run each candidate model on the input.
    4. Save outputs.
    5. Score each output with one or more metrics.
    6. Look at the numbers AND read the actual outputs. Numbers lie sometimes.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from rouge_score import rouge_scorer

load_dotenv()

ARTICLE = (
    "The Apollo program was a series of crewed spaceflight missions launched "
    "by NASA between 1961 and 1972 with the goal of landing humans on the "
    "Moon. Apollo 11, in July 1969, achieved the first crewed lunar landing, "
    "with astronauts Neil Armstrong and Buzz Aldrin walking on the surface "
    "while Michael Collins orbited above. The program ultimately included six "
    "successful landings and returned 382 kilograms of lunar samples."
)
PROMPT = f"Summarize the following in one sentence:\n\n{ARTICLE}"
REFERENCE = (
    "NASA's Apollo program (1961-1972) landed humans on the Moon, achieving "
    "six successful landings starting with Apollo 11 in 1969."
)

OUTPUTS_FILE = Path(__file__).parent / "outputs.json"


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


# ---- 1. Generate ------------------------------------------------------
outputs = {}
for label, fn in [("gemini-2.5-flash", from_gemini), ("llama-3.1-8b-hf", from_hf)]:
    try:
        outputs[label] = fn()
        print(f"[{label}] generated.")
    except Exception as e:
        outputs[label] = f"ERROR: {e}"
        print(f"[{label}] FAILED: {e}")

# ---- 2. Persist (so the eval is repeatable without re-spending API $$) ----
OUTPUTS_FILE.write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
print(f"\nOutputs saved to {OUTPUTS_FILE.name}")

# ---- 3. Score with ROUGE ---------------------------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
print(f"\n{'Model':<22} {'rouge1-F':>10} {'rouge2-F':>10} {'rougeL-F':>10}")
print("-" * 56)
for label, text in outputs.items():
    if text.startswith("ERROR:"):
        print(f"{label:<22} -- {text}")
        continue
    s = scorer.score(REFERENCE, text)
    print(
        f"{label:<22} {s['rouge1'].fmeasure:>10.3f} "
        f"{s['rouge2'].fmeasure:>10.3f} {s['rougeL'].fmeasure:>10.3f}"
    )

print("\nReference:")
print("  " + REFERENCE)
print("\nGenerated:")
for label, text in outputs.items():
    print(f"  [{label}] {text[:200]}")

print(
    "\nReminder: ROUGE only counts shared words. The model with the highest\n"
    "ROUGE may not be the most accurate or readable. Always read the outputs."
)
