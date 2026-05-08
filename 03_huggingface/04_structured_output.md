# `04_structured_output.py` — Concept Guide

## WHY (purpose)

The same need as `02_gemini/04_structured_json.py`: get JSON that conforms
to a schema, every call, with no string munging. The HF route is worth
seeing on its own because:

* It uses the **OpenAI-compatible `response_format` spec**, so the same
  block of code works against OpenAI, HF Inference, vLLM with an
  OpenAI-compatible server, and most third-party providers.
* Enforcement strictness varies by provider/model. So the *defensive*
  habit — validate again on your side with Pydantic — matters more here
  than it does on Gemini.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **OpenAI `response_format`** | A spec for asking models to return JSON. `type: "json_object"` (free-form JSON), `type: "json_schema"` (constrained to a schema). Adopted by HF and many other providers. |
| **`json_schema` block** | `{ "name": "...", "schema": <jsonschema dict>, "strict": true }`. `strict` asks the server to reject any output that doesn't validate. Not all providers honour `strict`. |
| **Pydantic → JSON Schema** | `MyModel.model_json_schema()` returns a dict in JSON-Schema-Draft form. Pydantic generates the schema automatically from your typed class — no hand-writing. |
| **Defensive parsing** | Even when the server claims to enforce the schema, validate again locally. `BaseModel.model_validate_json(raw)` parses + validates; both `json.JSONDecodeError` and `pydantic.ValidationError` should be handled. |
| **Why "system: only valid JSON"** | A safety net for models/providers that *don't* enforce strictly. Cheap insurance — it costs you a few extra tokens. |

## HOW (code walkthrough)

```python
from pydantic import BaseModel, ValidationError
from typing import List

class Book(BaseModel):
    title: str; author: str; year: int; genre: str

class BookList(BaseModel):
    books: List[Book]

schema = BookList.model_json_schema()
```

Pydantic does the schema work. You never hand-author JSON Schema.

```python
response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You return only valid JSON. No prose."},
        {"role": "user",   "content": "List three classic science fiction novels."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "BookList", "schema": schema, "strict": True},
    },
    max_tokens=400,
)
raw = response.choices[0].message.content
```

`response_format` is the OpenAI spec. The `system` message is the safety
net. `raw` is a JSON string — even with strict enforcement, treat it as
text until you've validated.

```python
try:
    parsed = BookList.model_validate_json(raw)
    for b in parsed.books:
        print(f"  {b.year}  {b.title!r}  by {b.author}  [{b.genre}]")
except (json.JSONDecodeError, ValidationError) as e:
    print(f"Validation failed: {e}")
```

Catch *both* exceptions. JSON-decode errors mean the model emitted
non-JSON; validation errors mean it emitted JSON of the wrong shape. In
production, log the raw text and the prompt — that's how you debug the
prompt later.

**Run it:**

```bash
python 03_huggingface/04_structured_output.py
```

Try changing the model to one whose provider doesn't honour `strict`.
You may suddenly see prose creep in — and the local Pydantic validation
catches it. This is exactly why defensive parsing belongs in the recipe,
not as a "polish later" item.
