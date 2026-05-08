# `04_structured_json.py` — Concept Guide

## WHY (purpose)

Production code does not consume English. It consumes records. The moment
an LLM output feeds a database, an API call, or a UI component, you need
the response in a strict shape — and you need that shape *guaranteed*, not
hoped for.

The naive approach ("just say 'reply in JSON' in the prompt") fails
intermittently: the model adds a preamble, wraps the JSON in ` ```json `
fences, omits a field, or invents one. Schema-constrained decoding moves
the guarantee from your regex to the model itself.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Structured output / constrained decoding** | The model is forced to emit tokens that conform to a JSON schema. Invalid tokens are masked out at generation time. Result: parseable JSON every call. |
| **`response_mime_type`** | Tells Gemini what format you want. `"application/json"` switches off the chatty preamble and returns raw JSON. By itself it does *not* enforce a schema — only the format. |
| **`response_schema`** | The schema. Accepts a Pydantic `BaseModel` class (recommended) or a plain dict in JSON Schema form. With this set, the JSON is guaranteed to validate against the schema. |
| **Pydantic `BaseModel`** | A Python class where each attribute has a type. The class doubles as: validation rules, JSON schema generator (`model_json_schema()`), and target type for parsing (`model_validate(...)`). |
| **`response.parsed`** | A Gemini convenience: when you supplied a Pydantic schema, the SDK already parsed the JSON into instances of your class. No `json.loads`, no `try/except`. |
| **Nested models** | Schemas compose: a `BookList` containing `List[Book]` becomes a JSON object with an array of objects. Use this to express any record shape. |

## HOW (code walkthrough)

```python
from pydantic import BaseModel
from typing import List

class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str

class BookList(BaseModel):
    books: List[Book]
```

The schema is just a Python type definition. Adding a field is a one-line
change; removing one is a one-line delete; changing a type narrows or
widens what the model is allowed to emit.

```python
from google.genai import types
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="List three classic science fiction novels. Return only structured data.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BookList,
    ),
)
```

Both flags matter. `response_mime_type` strips prose; `response_schema`
shapes the JSON. The prompt itself can be terse — the schema does the
heavy lifting.

```python
print(response.text)              # raw JSON string

parsed: BookList = response.parsed   # already a typed Python object
for book in parsed.books:
    print(f"  {book.year}  {book.title!r}  by {book.author}  [{book.genre}]")
```

`response.text` is the JSON string for logging or persistence.
`response.parsed` is the typed object for code paths that need
attribute access. In practice, prefer the parsed path — IDEs and type
checkers can then tell you when you mistype a field name.

**Run it:**

```bash
python 02_gemini/04_structured_json.py
```

Try adding a `summary: str` field to `Book` and rerun — the model fills
it in. Try changing `year: int` to `year: str` — observe how the model
adapts to the new type. The same Pydantic-class-as-schema pattern works
in tool definitions later on.
