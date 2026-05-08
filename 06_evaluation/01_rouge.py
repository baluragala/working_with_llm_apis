"""
06_evaluation/01_rouge.py
-----------------------------------------------------------------
ROUGE = Recall-Oriented Understudy for Gisting Evaluation.

Designed for summarization evaluation. Compares generated summary against
one or more human reference summaries.

The three flavors you'll see:
    rouge1  - unigram (single word) overlap
    rouge2  - bigram (two-word phrase) overlap
    rougeL  - longest common subsequence (rewards in-order overlap)

Each returns precision / recall / F1. Higher is better. ROUGE looks only
at surface word overlap, not meaning. A correct paraphrase scores low.
"""

from rouge_score import rouge_scorer

# ---- The article we asked the model to summarize --------------------
ARTICLE = (
    "The Apollo program was a series of crewed spaceflight missions launched "
    "by NASA between 1961 and 1972 with the goal of landing humans on the "
    "Moon. Apollo 11, in July 1969, achieved the first crewed lunar landing, "
    "with astronauts Neil Armstrong and Buzz Aldrin walking on the surface "
    "while Michael Collins orbited above. The program ultimately included six "
    "successful landings and returned 382 kilograms of lunar samples."
)

# ---- Human-written reference summary --------------------------------
REFERENCE = (
    "NASA's Apollo program (1961-1972) landed humans on the Moon. "
    "Apollo 11 made the first landing in 1969, and the program achieved "
    "six successful landings, returning 382 kg of lunar samples."
)

# ---- Two candidate model summaries ----------------------------------
CAND_A = (
    "Apollo was a NASA program from 1961 to 1972 that landed astronauts "
    "on the Moon, starting with Apollo 11 in 1969 and totaling six landings."
)
CAND_B = (
    "Some astronauts went to space in the 1960s and brought back rocks."
)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

for label, candidate in [("Candidate A", CAND_A), ("Candidate B", CAND_B)]:
    print(f"\n=== {label} ===")
    print(f"Text: {candidate}")
    scores = scorer.score(REFERENCE, candidate)
    for metric, s in scores.items():
        print(f"  {metric:8} P={s.precision:.3f}  R={s.recall:.3f}  F1={s.fmeasure:.3f}")

print(
    "\nInterpretation: A is closer to the reference (more shared content + ordering) "
    "so its F1 scores are higher. B is too vague to score well."
)
