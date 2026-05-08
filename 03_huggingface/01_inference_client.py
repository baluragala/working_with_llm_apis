"""
03_huggingface/01_inference_client.py
-----------------------------------------------------------------
Hugging Face's `InferenceClient` is one Python class that talks to many
inference providers (HF's own, plus Together, Fireworks, Replicate, etc.).

Three things to notice:
    1. The model is named "org/model-name" - same as Hub URLs.
    2. You can pin a `provider=` to control which backend serves the model.
    3. The chat_completion API mirrors OpenAI's: same messages format,
       same role names ("system" / "user" / "assistant"). This is intentional.

Docs: https://huggingface.co/docs/huggingface_hub/guides/inference
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(token=os.environ["HF_TOKEN"])

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "Explain prompt engineering in two sentences."},
    ],
    max_tokens=200,
)

print("Response text:")
print(response.choices[0].message.content)

print("\nFinish reason :", response.choices[0].finish_reason)
print("Token usage   :", response.usage)
