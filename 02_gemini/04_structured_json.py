"""
02_gemini/04_structured_json.py
-----------------------------------------------------------------
"Just give me JSON" is a constant production need. Two approaches:

    A) response_mime_type="application/json"   - Model returns valid JSON,
                                                 you parse it. No schema enforced.

    B) response_schema=<pydantic class or dict> - Model returns JSON that
                                                 conforms to the schema.
                                                 Strongly recommended for
                                                 anything you'll feed downstream.

We use approach B with a Pydantic model. response.parsed gives you typed objects
directly - no json.loads, no regex stripping, no try/except guesswork.
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class Book(BaseModel):
    title: str
    author: str
    year: int
    genre: str


class BookList(BaseModel):
    books: List[Book]


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=(
        "List three classic science fiction novels. "
        "Return only structured data."
    ),
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BookList,
    ),
)

# Two ways to read the result:
print("--- Raw JSON text ---")
print(response.text)
print()

print("--- Parsed Python object ---")
parsed: BookList = response.parsed
for book in parsed.books:
    print(f"  {book.year}  {book.title!r:35}  by {book.author}  [{book.genre}]")
