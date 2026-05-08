"""
04_local_model/02_compare_latency.py
-----------------------------------------------------------------
Same prompt, three backends, side-by-side timings.

This is meant for live demonstration. Numbers will vary wildly with:
    - your hardware
    - your network
    - current provider load
    - which models the provider has warm

Don't quote these numbers as benchmarks. The point is to show the SHAPE
of the trade-off, not absolute performance.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

PROMPT = "Explain prompt engineering in two sentences."


def time_call(label, fn):
    t0 = time.time()
    try:
        out = fn()
        dt = time.time() - t0
        print(f"\n[{label}] {dt:.2f}s")
        print(out.strip()[:300])
    except Exception as e:
        print(f"\n[{label}] FAILED: {type(e).__name__}: {e}")


# ---- Hosted: Gemini ---------------------------------------------------
def gemini_call():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model="gemini-2.5-flash", contents=PROMPT,
    )
    return r.text


# ---- Hosted: HF Inference --------------------------------------------
def hf_call():
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    r = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200,
    )
    return r.choices[0].message.content


# ---- Local: TinyLlama on this machine --------------------------------
def local_call():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


if __name__ == "__main__":
    print("Same prompt, three backends. First-run latency includes warm-up.\n")
    time_call("Gemini  ",        gemini_call)
    time_call("HF      ",        hf_call)
    time_call("Local TL",        local_call)
