"""
02_gemini/03_multimodal.py
-----------------------------------------------------------------
Gemini accepts more than text. The `contents` field can be a list mixing
text strings and Part objects (image bytes, file references, audio, video).

This script:
    1. Downloads a small public image (a cat).
    2. Sends it WITH a text question in the same request.
    3. Prints what the model said about the picture.

For local files, use Part.from_bytes(...) with the file bytes and mime_type.
For larger files (>20 MB) prefer the Files API: client.files.upload().
"""

import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Lorem Picsum serves a deterministic JPEG by ID — id=237 is a black labrador.
# We use it instead of e.g. Wikimedia because Wikimedia's UA policy now blocks
# generic clients, and picsum.photos has no such restriction.
IMAGE_URL = "https://picsum.photos/id/237/300/200.jpg"
resp = requests.get(IMAGE_URL, timeout=30)
resp.raise_for_status()
img_bytes = resp.content

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Describe this animal in one sentence, then list three traits as bullet points.",
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ],
)

print("Multimodal response:\n")
print(response.text)
