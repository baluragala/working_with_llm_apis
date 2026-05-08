"""
04_local_model/01_download_run_tinyllama.py
-----------------------------------------------------------------
Run a model on YOUR machine. No API call, no per-token cost, no network
after the initial download.

Model used: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - 1.1 billion parameters (small for a chat model)
    - ~2.2 GB on disk
    - Runs on CPU, but a GPU (or Apple Silicon MPS) is much faster

Trade-offs vs hosted APIs:
    + Privacy: nothing leaves your machine
    + No usage limits, predictable cost (electricity + your time)
    + Can run offline / air-gapped
    - Quality: a 1.1B model is far below GPT-4 / Gemini Flash
    - Hardware: bigger models need a real GPU and lots of VRAM
    - You manage updates, batching, scaling, monitoring yourself
    - No native tool calling for most small models

Auth note:
    Public models like TinyLlama need no token. Gated models (e.g., Llama,
    Gemma) require: (1) accepting terms on the model page, AND (2) a HF
    token with read access. A "read" token is enough; you don't need write.
"""

import os
import time
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()
# HF_TOKEN is only needed for gated models. TinyLlama is ungated.
hf_token = os.getenv("HF_TOKEN")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---- Pick the best device available -----------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"           # Apple Silicon
else:
    device = "cpu"
print(f"Using device: {device}")

# ---- Download (cached after first run) --------------------------------
# First run: pulls ~2 GB. Subsequent runs use the cache (~/.cache/huggingface).
print(f"\nLoading {MODEL_ID} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=hf_token,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)
print(f"Loaded in {time.time() - t0:.1f}s")

# ---- Format the prompt with the model's chat template -----------------
# Each chat model expects a specific format (special tokens for roles, etc.).
# apply_chat_template handles this for you given a list of messages.
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Explain prompt engineering in two sentences."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# ---- Generate ---------------------------------------------------------
inputs = tokenizer(prompt, return_tensors="pt").to(device)

t0 = time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
elapsed = time.time() - t0

# Strip the prompt tokens; keep only the model's reply.
new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\n--- Reply ({len(new_tokens)} tokens in {elapsed:.1f}s, "
      f"{len(new_tokens)/elapsed:.1f} tok/s) ---")
print(reply)
