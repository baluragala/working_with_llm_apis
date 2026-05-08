"""
03_huggingface/03_provider_selection.py
-----------------------------------------------------------------
"Inference Providers" is HF's clever twist: one client, many backends.

Why you care:
    * Same code can hit HF's own inference, Together AI, Fireworks, Replicate,
      Cerebras, Hyperbolic and more by changing one parameter.
    * You can route closed models too (e.g., several providers host Llama-3,
      DeepSeek, Qwen, etc., often cheaper or faster than the original vendor).
    * Avoids vendor lock-in: switch providers without rewriting your app.

This script does two things:
    1. Lists which providers currently serve a given model.
    2. Calls the model on two different providers and prints both replies.

If a provider isn't available for your account, you'll get an error - skip it.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, model_info

load_dotenv()
token = os.environ["HF_TOKEN"]

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ---- 1. Discover providers for this model -----------------------------
# model_info(...).inference_provider_mapping shows which providers host it.
try:
    info = model_info(MODEL, expand=["inferenceProviderMapping"], token=token)
    providers = list((info.inference_provider_mapping or {}).keys())
    print(f"Providers hosting {MODEL}:")
    for p in providers:
        print(f"  - {p}")
except Exception as e:
    print(f"Could not list providers: {e}")
    providers = []

print()

# ---- 2. Call the same model on two different providers ----------------
candidates = ["auto"] + providers[:2]   # "auto" lets HF choose for you
prompt = "In one sentence, why is Python popular?"

for provider in candidates:
    try:
        client = InferenceClient(token=token, provider=provider)
        resp = client.chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        print(f"[{provider:>14}] {text}")
    except Exception as e:
        print(f"[{provider:>14}] FAIL: {type(e).__name__}: {e}")
