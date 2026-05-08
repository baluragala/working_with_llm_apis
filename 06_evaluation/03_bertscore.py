"""
06_evaluation/03_bertscore.py
-----------------------------------------------------------------
BERTScore goes beyond n-grams. It embeds each token using a transformer
(typically RoBERTa) and matches tokens by cosine similarity in vector space.

Why this matters:
    "The cat sat on the mat."   <-> "A feline rested on a rug."
    BLEU/ROUGE: near zero (no shared words)
    BERTScore : high (the meaning is the same)

The cost is real: it loads a transformer model on first call (~500 MB).
First run is slow; subsequent runs use the cache.
"""

from bert_score import score

REFERENCE = (
    "NASA's Apollo program landed humans on the Moon between 1969 and 1972, "
    "with six successful crewed lunar landings."
)

CAND_PARAPHRASE = (
    "Between 1969 and 1972, the United States space agency successfully "
    "sent astronauts to the Moon's surface six times."
)
CAND_OVERLAP = (
    "The Apollo program of NASA achieved six lunar landings between 1969 "
    "and 1972."
)
CAND_OFFTOPIC = (
    "Pizza is a popular Italian dish made with dough, tomatoes, and cheese."
)

candidates = [CAND_PARAPHRASE, CAND_OVERLAP, CAND_OFFTOPIC]
references = [REFERENCE] * len(candidates)

# `score` returns precision, recall, F1 as torch tensors (one entry per pair).
P, R, F1 = score(
    candidates, references,
    lang="en", model_type="roberta-large", verbose=False,
)

labels = ["Paraphrase", "Word-overlap", "Off-topic"]
print(f"{'':14} {'P':>6} {'R':>6} {'F1':>6}")
for lab, p, r, f in zip(labels, P.tolist(), R.tolist(), F1.tolist()):
    print(f"{lab:14} {p:6.3f} {r:6.3f} {f:6.3f}")

print(
    "\nInterpretation:\n"
    "  Paraphrase scores high even though it shares few words with the reference.\n"
    "  Word-overlap scores slightly higher (closer wording).\n"
    "  Off-topic correctly drops sharply."
)
