"""
03_huggingface/05_other_tasks.py
-----------------------------------------------------------------
Hugging Face hosts more than chat models. The same InferenceClient exposes
task-specific endpoints. A few high-value ones for production work:

    summarization         - distill long text into short text
    text_classification   - sentiment, topic, intent
    translation           - one language to another
    feature_extraction    - embeddings (vectors) for similarity / RAG
    image_classification  - label an image
    automatic_speech_recognition (ASR) - audio -> text

Each task uses a model trained for it. Don't pass an LLM to summarization()
and a tiny BERT model to chat_completion() and expect either to work.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
client = InferenceClient(token=os.environ["HF_TOKEN"])

LONG_TEXT = (
    "The Apollo program was a series of missions launched by NASA between "
    "1961 and 1972 with the goal of landing humans on the Moon. Apollo 11, "
    "in July 1969, achieved the first crewed lunar landing, with astronauts "
    "Neil Armstrong and Buzz Aldrin walking on the surface while Michael "
    "Collins orbited above. The program ultimately included six successful "
    "landings, returned 382 kilograms of lunar samples, and demonstrated "
    "American technological capability during the Cold War."
)

# ---- Summarization ----------------------------------------------------
print("=== SUMMARIZATION ===")
try:
    s = client.summarization(LONG_TEXT, model="facebook/bart-large-cnn")
    print(s.summary_text if hasattr(s, "summary_text") else s)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

# ---- Text classification (sentiment) ----------------------------------
print("\n=== SENTIMENT CLASSIFICATION ===")
try:
    r = client.text_classification(
        "The new menu is overpriced but the dessert was incredible.",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    for item in r:
        print(f"  {item.label:10}  score={item.score:.3f}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

# ---- Embeddings -------------------------------------------------------
print("\n=== EMBEDDINGS (feature extraction) ===")
try:
    vec = client.feature_extraction(
        "Cats are adorable.",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    # vec is a numpy-like 2D array; show its shape and first 5 numbers.
    print(f"Vector shape: {vec.shape}")
    print(f"First 5 dims: {vec.flatten()[:5].tolist()}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
