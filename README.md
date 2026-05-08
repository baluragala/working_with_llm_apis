# Working with LLM APIs — Live Session Code

Companion code for the C8-W2-S1 live session (180-min workshop, run as 150 min).
All scripts are runnable end-to-end. Each one stands alone — no inter-file imports.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp 01_secure_setup/.env.example .env
# Open .env and paste your real keys
```

## Get your keys

| Provider | Where | What you need |
|----------|-------|----------------|
| Google Gemini | https://aistudio.google.com/apikey | `GEMINI_API_KEY` (free tier available) |
| Hugging Face | https://huggingface.co/settings/tokens | `HF_TOKEN` (read-access fine for most demos) |
| OpenAI | https://platform.openai.com/api-keys | `OPENAI_API_KEY` (requires billing setup) |

## Folder map

```
01_secure_setup/   .env handling, connectivity test
02_gemini/         minimal → params → multimodal → JSON → metadata → chat → tools
03_huggingface/    InferenceClient, providers, structured output, other tasks
04_local_model/    TinyLlama download + run + latency comparison
05_openai/         minimal example + tool calling (delta view)
06_evaluation/     ROUGE / BLEU / BERTScore + cross-model comparison
```

## Run order during the session

Follow the numeric prefix on filenames inside each folder. Every script prints what
it sent and what it got back, so it doubles as a teaching aid on screen.

## A note on costs

- Gemini and Hugging Face have free tiers that comfortably cover this session.
- OpenAI calls in this repo are tiny (`gpt-4o-mini`, ~1¢ total).
- The local model downloads ~2 GB on first run — do this before class if your
  network is slow.
