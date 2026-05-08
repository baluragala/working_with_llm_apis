# `05_other_tasks.py` — Concept Guide

## WHY (purpose)

LLMs are not the only model worth using. Plenty of production needs are
better — cheaper, faster, more accurate — served by smaller task-specific
models:

* Sentiment classification on millions of reviews → a 100M-parameter
  fine-tuned classifier beats GPT-4 on cost by orders of magnitude.
* Embeddings for RAG → a 22M-parameter sentence-transformer is the
  industry default; using an LLM here is wasteful.
* Summarisation of news articles → a fine-tuned BART is fast and
  predictable.

The same `InferenceClient` exposes all of these via task-specific
methods. This script tours the most useful ones so learners know what's
on the shelf before reaching for an LLM.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Task-specific endpoints** | `summarization()`, `text_classification()`, `translation()`, `feature_extraction()`, `image_classification()`, `automatic_speech_recognition()`, plus more. Each is backed by a model trained for that task. |
| **Model choice = task choice** | Each task expects a model trained for it. Passing an LLM to `summarization()` or a tiny BERT to `chat_completion()` produces nonsense or errors. |
| **Summarisation** | Distil long text into short text. Encoder-decoder models like `facebook/bart-large-cnn` (CNN/DailyMail-trained) are the canonical pick. |
| **Text classification** | Map text → label(s) with confidence. Sentiment is the most common variant; intent classification, topic tagging, toxicity detection are all the same shape. |
| **Embeddings (`feature_extraction`)** | Map text → fixed-size vector that captures meaning. Used for similarity search, clustering, RAG retrieval. `sentence-transformers/all-MiniLM-L6-v2` is the workhorse default — small, fast, decent. |
| **Cost/latency profile** | These models are 10–100× smaller than chat LLMs, so per-call cost and latency are typically far lower. For a high-volume pipeline this dominates the architectural choice. |

## HOW (code walkthrough)

```python
client = InferenceClient(token=os.environ["HF_TOKEN"])
LONG_TEXT = "...Apollo program..."
```

One client, three different tasks below.

```python
s = client.summarization(LONG_TEXT, model="facebook/bart-large-cnn")
print(s.summary_text if hasattr(s, "summary_text") else s)
```

A single line for summarisation. The return type is a small structured
object with `summary_text`. The `hasattr` guard accommodates older library
versions that returned a plain dict.

```python
r = client.text_classification(
    "The new menu is overpriced but the dessert was incredible.",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
for item in r:
    print(f"  {item.label:10}  score={item.score:.3f}")
```

Returns a list of `(label, score)` candidates ranked by confidence. SST-2
emits `POSITIVE` / `NEGATIVE`. For multi-class needs, swap in a model
fine-tuned for those labels — same call, same return shape.

```python
vec = client.feature_extraction(
    "Cats are adorable.",
    model="sentence-transformers/all-MiniLM-L6-v2",
)
print(f"Vector shape: {vec.shape}")
print(f"First 5 dims: {vec.flatten()[:5].tolist()}")
```

The result is a numpy-like array. For `MiniLM-L6-v2` it's 384-dimensional.
Plug this output into FAISS / pgvector / Pinecone for similarity search;
that is the foundation of RAG.

**Run it:**

```bash
python 03_huggingface/05_other_tasks.py
```

Notice the response speed compared with the chat completions earlier in
this folder. The lesson: reach for an LLM when you need open-ended
reasoning; reach for a task-specific model when the task fits a
well-trodden category.
