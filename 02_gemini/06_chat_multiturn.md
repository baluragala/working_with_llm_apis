# `06_chat_multiturn.py` — Concept Guide

## WHY (purpose)

LLMs are stateless. The model has no memory between calls — every request
starts from scratch as far as the model is concerned. Yet every chat
product behaves as if it remembers the conversation.

The trick: the *application* keeps the history, and replays the whole
conversation on every turn. This script demonstrates two ways to do that
with Gemini — a manual list (full control, useful when you need to
trim/edit/persist history yourself) and the `chats` helper (less code,
fine for typical use).

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Statelessness of LLM APIs** | Each `generate_content` call is independent. The server keeps no per-user memory across calls (some products simulate it, but at the API level it's stateless). |
| **Conversation history as input** | The "memory" is just a list of past turns prepended to each request. As history grows, so does the prompt — eventually you have to summarise or trim. |
| **Roles in Gemini** | `"user"` for the human, `"model"` for the assistant. (OpenAI / HuggingFace use `"assistant"` for the same thing.) System prompts are a separate config field, not a role. |
| **`types.Content` and `types.Part`** | A `Content` is one turn (role + parts); a `Part` is a chunk inside a turn (text, image, function call, function response). A turn can have multiple parts — e.g. text + image. |
| **`client.chats.create`** | A higher-level helper that owns the history list internally. `chat.send_message(text)` appends the user turn, calls the model, appends the model turn, returns the response. |
| **`system_instruction` in chat sessions** | Set once on the chat session and applied to every turn. No need to repeat it in history. |
| **`chat.get_history()`** | Returns the full list of `Content` objects the helper has accumulated. Useful for persistence, debugging, or migrating to manual mode. |

## HOW (code walkthrough)

### Approach 1 — manual history

```python
history = [
    types.Content(role="user", parts=[types.Part(text="My name is Asha. Remember it.")]),
]

r1 = client.models.generate_content(model="gemini-2.5-flash", contents=history)
history.append(types.Content(role="model", parts=[types.Part(text=r1.text)]))

history.append(types.Content(role="user", parts=[types.Part(text="What is my name?")]))
r2 = client.models.generate_content(model="gemini-2.5-flash", contents=history)
```

The pattern is always: append the user turn, call the model, append the
model turn. Pass `history` as `contents` — that's how the model "remembers".
You own the list, so you decide what to keep, drop, or summarise as the
conversation grows.

### Approach 2 — `chats` helper

```python
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a concise assistant. Keep answers under 30 words.",
    ),
)

reply = chat.send_message("I'm planning a 3-day trip to Goa.")
reply = chat.send_message("Suggest one beach for day 1.")
reply = chat.send_message("And what about food there?")

print(len(chat.get_history()), "turns in memory")
```

The helper hides the history bookkeeping. Use it for the 90% case. Drop
back to manual mode when you need to trim, edit past turns, or persist
history to your own storage between sessions.

**Run it:**

```bash
python 02_gemini/06_chat_multiturn.py
```

Comment out the line that appends the model's reply in approach 1 and
rerun — the model now answers "What is my name?" with "I don't know".
That single deletion is the difference between a chatbot that remembers
and one that doesn't, and shows that "memory" is *just* the history list.
