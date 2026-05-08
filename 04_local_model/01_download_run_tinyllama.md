# `01_download_run_tinyllama.py` — Concept Guide

## WHY (purpose)

Until now everything has been a remote API call. Two situations push you
toward running the model on your own hardware:

* **Privacy.** Healthcare records, internal source code, customer PII —
  there are domains where shipping the prompt to a third party is a
  non-starter.
* **Predictable cost.** API providers bill per token. Run-it-yourself
  bills electricity. For high-volume, narrow workloads, the math often
  flips after a few months.

The trade-off is quality and operational burden: a 1.1B-parameter local
model is meaningfully worse than Gemini Flash, and you now own deployment,
patching, scaling, and monitoring. This script makes those trade-offs
concrete by doing one full local generation end-to-end.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **`transformers`** | Hugging Face's PyTorch library. Two key classes: `AutoTokenizer` (text ↔ token ids) and `AutoModelForCausalLM` (the model itself). |
| **TinyLlama** | A 1.1B-parameter chat-tuned model. ~2.2 GB on disk, runs on CPU, fast on a GPU or Apple-Silicon MPS. Quality is far below GPT-4 / Gemini — it's a *learning* model, not a production one. |
| **HF cache** | First call to `from_pretrained(...)` downloads weights to `~/.cache/huggingface`. Subsequent calls load from disk. |
| **Gated models** | Some models (Llama, Gemma, Mistral variants) require accepting terms on the model page **plus** a read-scope HF token. TinyLlama is *un*gated, so the token is optional. |
| **Device dispatch** | PyTorch runs on `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu` (always). Detect and pick the best one available. |
| **`torch_dtype`** | Numerical precision of weights. `float16` (or `bfloat16`) is half the memory and faster on GPU/MPS; `float32` is required on most CPUs. |
| **Chat template** | Each chat model expects a model-specific prompt format with role markers (e.g. `<|user|>`, `<|assistant|>`). `tokenizer.apply_chat_template(messages, ...)` turns a `messages` list into the right string for that model. |
| **`add_generation_prompt=True`** | Tells the template to leave the trailing `<|assistant|>` marker so the model knows to start generating in that role. |
| **`model.generate(...)`** | The generation loop. `max_new_tokens` (length cap), `do_sample` (sampling vs greedy), `temperature` / `top_p` (same meaning as in APIs), `pad_token_id` (avoid a warning when the model has no pad token). |
| **`torch.no_grad()`** | Disables gradient tracking. Inference doesn't need gradients; this saves memory and a bit of time. |
| **Stripping the prompt from the output** | `model.generate` returns prompt + completion tokens. Slice off the prompt before decoding so you only print what the model actually said. |

## HOW (code walkthrough)

```python
device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
```

Pick the best available accelerator. On an M-series Mac without a CUDA
GPU, `mps` is roughly 5–10× faster than CPU.

```python
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=hf_token,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)
```

The first run downloads ~2 GB; later runs read from cache. `token` is
passed even though TinyLlama is ungated — this also raises rate limits.

```python
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Explain prompt engineering in two sentences."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
```

`apply_chat_template` is the bridge between the OpenAI-shaped `messages`
list and the model-specific token format. Skipping this step is the
single most common source of "the model ignores the system prompt"
complaints in local serving.

```python
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True, temperature=0.7, top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
```

The slice `output_ids[0][prompt_len:]` keeps only the generated tokens.
`skip_special_tokens=True` drops `<|user|>` / `<|assistant|>` markers so
you see clean text.

**Run it:**

```bash
python 04_local_model/01_download_run_tinyllama.py
```

The first run is dominated by the download. Subsequent runs are dominated
by inference. Compare the response text to the same prompt run through
Gemini Flash or Llama-3.1-8B (the HF examples) — quality and verbosity
will differ noticeably. That difference *is* the lesson.
