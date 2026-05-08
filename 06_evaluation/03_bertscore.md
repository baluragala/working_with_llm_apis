# `03_bertscore.py` — Concept Guide

## WHY (purpose)

ROUGE and BLEU agree on one thing: a paraphrase that uses synonyms is
indistinguishable from a wrong answer. That's a serious problem in 2026
when modern LLMs paraphrase competently — these metrics systematically
under-score them.

BERTScore fixes the diagnosis. Instead of comparing words, it compares
**meanings** by embedding each token with a transformer and matching
tokens by cosine similarity. A paraphrase that swaps "NASA" for "the
United States space agency" still scores high.

This script puts a paraphrase, a near-overlap, and an off-topic candidate
through BERTScore so the contrast is unmistakable.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **BERTScore** | A semantic similarity metric. For each candidate token, find the most similar reference token by cosine similarity in transformer-embedding space; aggregate to precision, recall, F1. |
| **Contextual embeddings** | Unlike static embeddings (word2vec), a transformer produces a different vector for the same word in different contexts. "bank" in "river bank" vs "bank account" gets different vectors — so BERTScore handles polysemy. |
| **`model_type="roberta-large"`** | The default high-quality choice for English. Other options trade quality for speed (`distilbert-*`) or cover more languages (`xlm-roberta-large`). |
| **Cost** | First run downloads a ~500 MB transformer to your HF cache. Subsequent runs are warm. Per-pair inference is far slower than ROUGE/BLEU — a few hundred ms per pair on CPU, sub-100 ms on GPU. |
| **Precision / Recall / F1** | Same triple as ROUGE. Precision = "every candidate token has a matching reference token". Recall = "every reference token is covered by some candidate token". F1 is the score you usually report. |
| **Caveats** | Sensitive to length and the choice of underlying model. Doesn't replace human eval for high-stakes judgements; complements it. Use a fixed `model_type` when comparing scores across runs. |

## HOW (code walkthrough)

```python
from bert_score import score

REFERENCE = "NASA's Apollo program landed humans on the Moon between 1969 and 1972..."
candidates = [
    "Between 1969 and 1972, the United States space agency successfully sent astronauts to the Moon's surface six times.",  # paraphrase
    "The Apollo program of NASA achieved six lunar landings between 1969 and 1972.",                                          # heavy word overlap
    "Pizza is a popular Italian dish made with dough, tomatoes, and cheese.",                                                  # off-topic
]
references = [REFERENCE] * len(candidates)

P, R, F1 = score(
    candidates, references,
    lang="en", model_type="roberta-large", verbose=False,
)
```

`score` takes parallel lists of candidates and references — pair `i` is
compared element-wise. It returns three torch tensors (one entry per
pair) for precision, recall, F1.

```python
labels = ["Paraphrase", "Word-overlap", "Off-topic"]
for lab, p, r, f in zip(labels, P.tolist(), R.tolist(), F1.tolist()):
    print(f"{lab:14} {p:6.3f} {r:6.3f} {f:6.3f}")
```

### How to read the numbers

* **Paraphrase**: high score. The metric correctly recognises that
  "United States space agency" ≈ "NASA". This is the thing ROUGE/BLEU
  can't do.
* **Word-overlap**: slightly higher than the paraphrase (it shares more
  surface words *and* preserves meaning).
* **Off-topic**: scores drop sharply — semantically distant content shows
  up as such.

**Run it:**

```bash
python 06_evaluation/03_bertscore.py
```

First run takes a while because of the model download. Future runs read
from `~/.cache/huggingface`. Use BERTScore as your "the meaning is right
even if the words are different" check, not as a single-number truth —
its cost is real, and it's still a metric, not a judgement.
