"""
05_openai/01_minimal.py
-----------------------------------------------------------------
The smallest possible OpenAI chat request, presented as a "delta view"
against Gemini and HF so learners can see the same idea three ways.

Same task as:
    02_gemini/01_minimal.py
    03_huggingface/02_chat_completion.py (the deterministic block)

Differences worth pointing out in class:

    GEMINI                         HUGGINGFACE                  OPENAI
    genai.Client(api_key=...)      InferenceClient(token=...)   OpenAI(api_key=...)
    client.models                  client.chat_completion(...)  client.chat.completions
        .generate_content(...)                                       .create(...)

    contents="..."                 messages=[{"role": "user",   messages=[{"role": "user",
    (string OR list of parts)        "content": "..."}]            "content": "..."}]

    roles: "user" / "model"        roles: system/user/assistant roles: system/user/assistant
    system instruction is          system prompt is a regular   system prompt is a regular
    a separate config field        message                      message

    response.text                  response.choices[0]          response.choices[0]
                                     .message.content             .message.content

The mental model is the same in all three: you send a message list (or a
prompt) to a model and get back a response object whose .content / .text
holds the generated text. Memorise the *shape*, not the SDK.

Docs: https://platform.openai.com/docs/api-reference/chat
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain prompt engineering in two sentences."},
    ],
)

print("Response text:")
print(response.choices[0].message.content)

# A peek at the response envelope - useful for the "metadata" discussion.
print("\nUsage:")
print(f"  prompt_tokens     = {response.usage.prompt_tokens}")
print(f"  completion_tokens = {response.usage.completion_tokens}")
print(f"  total_tokens      = {response.usage.total_tokens}")
print(f"  finish_reason     = {response.choices[0].finish_reason}")
