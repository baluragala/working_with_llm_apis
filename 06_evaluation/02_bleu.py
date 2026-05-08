"""
06_evaluation/02_bleu.py
-----------------------------------------------------------------
BLEU = Bilingual Evaluation Understudy.

Designed for translation. Like ROUGE, it counts n-gram overlap with one or
more references, but BLEU is precision-focused (with a brevity penalty so
you can't game it by emitting a single word).

Score range: 0 - 100 (sacrebleu) or 0.0 - 1.0 (NLTK). 30+ is good for MT.
BLEU on summarization is unusual; on translation it is the de-facto baseline.
"""

import sacrebleu

# Source: "Where is the railway station?" in English -> Hindi.
REFERENCES = [[
    "रेलवे स्टेशन कहाँ है?",
    "रेलवे स्टेशन कहाँ पर है?",
]]   # list-of-references (sacrebleu wants list of lists)

CAND_A = "रेलवे स्टेशन कहाँ है?"          # Matches reference 1 exactly
CAND_B = "रेलवे स्टेशन कहाँ पर है?"        # Matches reference 2 exactly
CAND_C = "ट्रेन का स्टेशन कहाँ है?"         # Synonym for railway, partial overlap
CAND_D = "मुझे चाय चाहिए।"                  # Wrong sentence entirely

for label, cand in [("A (exact match ref1)", CAND_A),
                    ("B (exact match ref2)", CAND_B),
                    ("C (synonym)         ", CAND_C),
                    ("D (off topic)       ", CAND_D)]:
    bleu = sacrebleu.sentence_bleu(cand, [r[0] for r in REFERENCES] + [REFERENCES[0][1]])
    print(f"{label}  BLEU = {bleu.score:5.2f}   {cand}")

print(
    "\nNote: BLEU rewards exact n-gram matches. Synonyms are punished even "
    "when they preserve meaning. Always pair BLEU with human or semantic eval."
)
