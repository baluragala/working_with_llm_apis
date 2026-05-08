# `05_metadata.py` — Concept Guide

## WHY (purpose)

`response.text` is the answer to "what did the model say?". Almost every
*other* operational question — "did it finish or get cut off?", "did it
trip a safety filter?", "how much did this cost?", "should I retry?" —
lives in the response object's metadata fields.

In a workshop you can ignore them. In a paid pipeline running thousands
of calls a day, ignoring them costs you money and quietly hides bugs.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **`candidates`** | A list of completions for the same prompt. With `candidate_count=1` (the default) you get one. Some models / tasks let you ask for more (e.g. n-best for re-ranking). |
| **`finish_reason`** | Why the model stopped. Common values: `STOP` (natural end), `MAX_TOKENS` (you hit the cap; output is truncated), `SAFETY` (content policy), `RECITATION`, `OTHER`. Always check; never assume. |
| **`safety_ratings`** | Per-category safety classifications: harassment, hate speech, sexually explicit, dangerous. Each has a `probability` bucket (NEGLIGIBLE / LOW / MEDIUM / HIGH). Useful for human-in-the-loop review queues. |
| **`usage_metadata`** | Token accounting: `prompt_token_count` (what you sent), `candidates_token_count` (what came back), `total_token_count`. This is what providers bill on. |
| **Tokens vs words** | Roughly 1 token ≈ ¾ of an English word, but it varies wildly with language and content. Token counts are the only honest cost signal. |
| **Why log it all** | If quality regresses next week, the first question is "did `finish_reason=MAX_TOKENS` rates spike?". You cannot answer if you only stored `.text`. |

## HOW (code walkthrough)

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Give me three reasons to learn Python.",
    config=types.GenerateContentConfig(
        max_output_tokens=200,
        candidate_count=1,
    ),
)
print(response.text)
```

Same call shape as `01_minimal.py`. The interesting bits are below.

```python
for i, cand in enumerate(response.candidates):
    print(f"Candidate {i}: finish_reason = {cand.finish_reason}")
    if cand.safety_ratings:
        for sr in cand.safety_ratings:
            print(f"  safety: {sr.category.name} -> {sr.probability.name}")
```

Iterating `response.candidates` is the right shape even when there's only
one — your code stays correct if you later raise `candidate_count`.
`finish_reason` and `safety_ratings` are enums; their `.name` attribute
gives a readable string for logs.

```python
u = response.usage_metadata
print(f"prompt_token_count     : {u.prompt_token_count}")
print(f"candidates_token_count : {u.candidates_token_count}")
print(f"total_token_count      : {u.total_token_count}")
```

The three numbers you want in every observability dashboard. Multiply by
the published per-token price to get a per-call cost.

**Run it:**

```bash
python 02_gemini/05_metadata.py
```

Then drop `max_output_tokens` to `20` and rerun — `finish_reason` flips
to `MAX_TOKENS` and the text is mid-sentence. That single field is what
distinguishes "the model gave a short answer" from "the model was cut off".
