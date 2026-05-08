# `01_minimal.py` — Concept Guide

## WHY (purpose)

Most teams talk to more than one provider. Treating each SDK as a separate
mental model is exhausting and error-prone. This script is the third
"hello world" — same task as `02_gemini/01_minimal.py` and
`03_huggingface/01_inference_client.py` — done in OpenAI's SDK so the
**delta** between vendors is staring the learner in the face.

Once you see the three side by side, the lesson is the same the workshop
keeps repeating: memorise the *shape* (client + messages + a method), not
the syntactic details of any one SDK.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **`openai` SDK** | The official Python client. Construct `OpenAI(api_key=...)` once and reuse. The same SDK targets OpenAI's endpoint by default and can be repointed at any OpenAI-compatible server (vLLM, OpenRouter, Together, Fireworks…). |
| **`client.chat.completions.create`** | The chat endpoint. Takes a `messages` list with `system`/`user`/`assistant` roles — same shape as Hugging Face's `chat_completion`. |
| **Model identifier** | A short string like `"gpt-4o-mini"` (cheap, fast) or `"gpt-4o"` (smarter). Distinct from Gemini (org-less names) and HF (org/name slugs). |
| **`response.choices[0].message.content`** | The reply text. The list-of-choices shape is identical to HF's. With `n>1` you get multiple completions for re-ranking. |
| **`response.usage.{prompt,completion,total}_tokens`** | The OpenAI version of Gemini's `usage_metadata`. Same three numbers, different attribute path. |
| **`finish_reason`** | Why generation stopped. Common values: `stop` (natural end), `length` (hit `max_tokens`), `content_filter`, `tool_calls`. Lowercase strings here vs Gemini's enum names. |

## HOW (code walkthrough)

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

Same construction pattern as the other two SDKs: build a client with a
credential, reuse the handle.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain prompt engineering in two sentences."},
    ],
)
print(response.choices[0].message.content)
```

The minimal call. No system prompt, no temperature, no tools — exactly
the same shape as the Gemini and HF "hello worlds", just with OpenAI's
attribute names.

```python
print(f"prompt_tokens     = {response.usage.prompt_tokens}")
print(f"completion_tokens = {response.usage.completion_tokens}")
print(f"total_tokens      = {response.usage.total_tokens}")
print(f"finish_reason     = {response.choices[0].finish_reason}")
```

A peek at the metadata to bridge with `02_gemini/05_metadata.py`. Same
fields, different names.

### The three-vendor delta in one table

| | Gemini | Hugging Face | OpenAI |
|---|---|---|---|
| Client | `genai.Client(api_key=...)` | `InferenceClient(token=...)` | `OpenAI(api_key=...)` |
| Method | `client.models.generate_content(...)` | `client.chat_completion(...)` | `client.chat.completions.create(...)` |
| Prompt arg | `contents="..."` (or list of parts) | `messages=[...]` | `messages=[...]` |
| Roles | `user` / `model` | `system` / `user` / `assistant` | `system` / `user` / `assistant` |
| System prompt | `config.system_instruction` | role `system` in messages | role `system` in messages |
| Reply text | `response.text` | `response.choices[0].message.content` | `response.choices[0].message.content` |
| Token usage | `response.usage_metadata.{prompt,candidates,total}_token_count` | `response.usage` | `response.usage.{prompt,completion,total}_tokens` |

**Run it:**

```bash
python 05_openai/01_minimal.py
```

If you've already run the Gemini and HF "hello worlds", you have three
text outputs that say roughly the same thing — and three bits of
boilerplate that look almost the same. That mental alignment is the
whole point of doing OpenAI third.
