# `03_multimodal.py` — Concept Guide

## WHY (purpose)

A surprising amount of real-world LLM work is "look at this image and
tell me X" — receipt parsing, screenshot QA, alt-text generation, insurance
claim photos, accessibility descriptions. These tasks need a model that
accepts images alongside text in the same request, not a separate
vision-only model wired up over the top.

Gemini is natively multimodal: `contents` can mix strings and binary parts
(images, PDFs, audio, video) in one call. Once you've sent one image, the
same pattern extends to the rest.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Multimodal input** | A request that mixes more than one modality. Here: text + image. The model jointly attends over both when generating. |
| **`contents` as a heterogeneous list** | Instead of a single string, you pass `[ "ask about this", Part(image_bytes) ]`. Order matters for some prompts (text before image vs after) but for simple Q&A either works. |
| **`types.Part`** | Gemini's wrapper around a single piece of content. `Part.from_bytes(data=..., mime_type=...)` builds an inline image part. Other constructors: `Part.from_uri`, `Part.from_function_response`, plain text. |
| **Inline bytes vs Files API** | Inline is fine up to ~20 MB total request. Beyond that, upload first with `client.files.upload(...)` and reference the returned handle. Audio and video almost always go through Files. |
| **MIME type matters** | The server uses `mime_type` to pick the decoder. Send `"image/jpeg"` for a JPEG, `"image/png"` for a PNG, `"application/pdf"` for a PDF. Wrong MIME → "Unable to process input" 400. |
| **HTTP download hygiene** | When you fetch the image yourself, set a User-Agent (some hosts reject default UAs) and call `raise_for_status()` so a 4xx surfaces *before* you hand bytes to the model. |

## HOW (code walkthrough)

```python
IMAGE_URL = "https://picsum.photos/id/237/300/200.jpg"
resp = requests.get(IMAGE_URL, timeout=30)
resp.raise_for_status()
img_bytes = resp.content
```

We use Lorem Picsum's deterministic-by-id endpoint (id 237 is a black
labrador) so the script is reproducible. `raise_for_status()` is critical
— without it, a non-200 body (e.g. an HTML error page) gets passed to
Gemini as JPEG bytes, and the model returns a confusing 400.

```python
from google.genai import types
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Describe this animal in one sentence, then list three traits as bullet points.",
        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ],
)
print(response.text)
```

The text and the image are siblings in the list. The model sees both and
answers the text question grounded on the image. To swap to a local file,
replace the `requests.get(...).content` with `open(path, "rb").read()` —
the rest is unchanged.

**Run it:**

```bash
python 02_gemini/03_multimodal.py
```

You should get a one-sentence description of a dog plus three bullets.
Try replacing the URL with a screenshot of a receipt and asking
"return the line items as JSON" — same code, very different application.
