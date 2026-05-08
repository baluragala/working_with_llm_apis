# `01_rouge.py` — Concept Guide

## WHY (purpose)

"Did the model produce a good summary?" is hard to answer in code. ROUGE
is the historical, almost universal first answer for summarisation
evaluation. It's far from perfect — it can't tell you a paraphrase is
correct — but it's cheap, reproducible, and well-understood, which is
exactly what you want as a regression metric.

This script shows the metric working on two candidate summaries, one
faithful and one vague, so the score interpretation is concrete.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation. A family of n-gram overlap metrics for comparing a candidate text to one or more reference texts. Purely lexical — looks at word surface, not meaning. |
| **`rouge1`** | Unigram (single-word) overlap. Captures topic / content overlap. Insensitive to word order. |
| **`rouge2`** | Bigram (two-word phrase) overlap. Sensitive to local phrasing. Lower scores than rouge1 for the same pair. |
| **`rougeL`** | Longest Common Subsequence based. Rewards in-order overlap, even with insertions between matched words. Good proxy for "did the candidate preserve the structure of the reference?" |
| **Precision / Recall / F1** | Per metric: precision = % of candidate n-grams found in reference; recall = % of reference n-grams found in candidate; F1 = harmonic mean. F1 is the score you usually report. |
| **Stemming (`use_stemmer=True`)** | "running" and "run" map to the same stem. Avoids penalising trivial morphological differences. Standard for English ROUGE. |
| **Known weakness** | A correct paraphrase that uses different vocabulary scores low. Always pair ROUGE with at least one semantic metric (BERTScore) or human judgement on critical paths. |

## HOW (code walkthrough)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
)
```

`RougeScorer` is reusable across many candidate/reference pairs. Build it
once.

```python
REFERENCE = "NASA's Apollo program (1961-1972) landed humans on the Moon..."

CAND_A = "Apollo was a NASA program from 1961 to 1972 that landed astronauts..."
CAND_B = "Some astronauts went to space in the 1960s and brought back rocks."

for label, candidate in [("Candidate A", CAND_A), ("Candidate B", CAND_B)]:
    scores = scorer.score(REFERENCE, candidate)
    for metric, s in scores.items():
        print(f"{metric:8} P={s.precision:.3f} R={s.recall:.3f} F1={s.fmeasure:.3f}")
```

`scorer.score(reference, candidate)` (note the order — reference first)
returns a dict keyed by metric name. Each value has `precision`, `recall`,
`fmeasure` attributes.

### How to read the numbers

* Candidate A scores higher across the board because it shares more
  unigrams, more bigrams, and a longer common subsequence with the
  reference. The summary is faithful and uses similar vocabulary.
* Candidate B scores low because it's vague — it picks up almost no
  bigrams and a poor longest common subsequence. The metric reflects
  that, even though a human reading both would also rate B as worse.

**Run it:**

```bash
python 06_evaluation/01_rouge.py
```

Try a third candidate that is *correct but uses different words* — e.g.,
"Between 1961 and 1972 the United States space agency completed crewed
lunar landings, returning hundreds of kilos of moon rock." Watch the
score drop even though the meaning is preserved. That collapse is
exactly the limitation that motivates `03_bertscore.py`.
