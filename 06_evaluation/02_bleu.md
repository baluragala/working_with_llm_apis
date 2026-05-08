# `02_bleu.py` — Concept Guide

## WHY (purpose)

BLEU is the de-facto baseline for machine-translation evaluation, and the
de-facto wrong choice for almost everything else. The reason it's still
in this workshop is twofold:

1. You will encounter BLEU in literature, leaderboards, and product
   reviews — being able to read it matters.
2. It's the cleanest illustration of an **n-gram-overlap-with-brevity-
   penalty** metric, and watching it fail on synonyms drives home why
   semantic metrics exist.

The script scores four Hindi candidate translations against two valid
references so the score gradient is obvious.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **BLEU** | Bilingual Evaluation Understudy. Counts n-gram precision (how many of the candidate's n-grams appear in any reference) for n = 1..4, geometric-means them, then multiplies by a brevity penalty. |
| **Brevity penalty** | A multiplier ≤ 1 that punishes candidates shorter than the reference. Without it, "the" would score 100% precision on most references. |
| **Multiple references** | BLEU was designed for cases where multiple correct translations exist. The candidate's n-grams need only appear in *any one* of them. |
| **Score range** | Modern tools (`sacrebleu`) report 0–100. NLTK reports 0.0–1.0. The library matters when you compare numbers across papers. **30+ is generally considered good** for machine translation. |
| **`sacrebleu`** | The community-standard implementation. Designed to be deterministic and tokenisation-stable so numbers are comparable across teams. |
| **Limitation** | Pure n-gram precision: a perfect-meaning paraphrase using different vocabulary scores poorly. This is the same lexical-blindness problem ROUGE has. |

## HOW (code walkthrough)

```python
import sacrebleu

REFERENCES = [[
    "रेलवे स्टेशन कहाँ है?",
    "रेलवे स्टेशन कहाँ पर है?",
]]   # list of references (sacrebleu wants list-of-lists)

CAND_A = "रेलवे स्टेशन कहाँ है?"          # exact match ref 1
CAND_B = "रेलवे स्टेशन कहाँ पर है?"        # exact match ref 2
CAND_C = "ट्रेन का स्टेशन कहाँ है?"         # synonym for "railway"
CAND_D = "मुझे चाय चाहिए।"                  # off-topic
```

Two valid references for one English source. The four candidates span the
spectrum: exact match, exact match (different reference), synonymous, and
unrelated.

```python
for label, cand in [...]:
    bleu = sacrebleu.sentence_bleu(cand, [r[0] for r in REFERENCES] + [REFERENCES[0][1]])
    print(f"{label}  BLEU = {bleu.score:5.2f}   {cand}")
```

`sentence_bleu(candidate, references)` computes BLEU for a single sentence
against multiple references. The flattening on the references is just to
match `sacrebleu`'s expected input shape.

### How to read the numbers

* A and B score very high — they each match a reference token-for-token.
* C scores poorly even though it preserves the meaning ("ट्रेन का
  स्टेशन" = "the train's station" vs the reference's "railway station").
  Different vocabulary, same idea — BLEU can't tell.
* D scores near zero — different words, different meaning.

**Run it:**

```bash
python 06_evaluation/02_bleu.py
```

The C-vs-D gap is the lesson. A correct paraphrase and a wrong sentence
both get low BLEU; the metric does not reward you for being right with
different words. That is why translation teams pair BLEU with chrF, COMET,
or human judgement, and it's why the next script (BERTScore) exists.
