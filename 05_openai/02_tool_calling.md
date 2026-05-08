# `02_tool_calling.py` — Concept Guide

## WHY (purpose)

`02_gemini/07_tool_calling.py` proved the conceptual shape of tool calling
on Gemini. This script ports the *exact same* weather example to OpenAI's
SDK so learners see that:

* The flow is identical: declare tool → model proposes call → you execute
  → you send the result back → model finalises the answer.
* The wrappers are different: OpenAI uses plain dicts (JSON Schema all
  the way), a `tool_calls` list on the assistant message, and a `tool`
  role for results — versus Gemini's `Tool` / `FunctionDeclaration` /
  `Part.from_function_response` objects.

Internalising the shared flow lets you read either provider's docs
without re-learning the model.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **OpenAI tool spec** | A list of `{"type": "function", "function": {name, description, parameters}}` dicts. `parameters` is a plain JSON Schema. No special wrapper classes. |
| **`tool_calls` on the assistant message** | When the model decides to use a tool, `response.choices[0].message.tool_calls` is populated; `content` is `None`. Each entry has a unique `id`, the function `name`, and `arguments` as a JSON *string* (not parsed). |
| **`json.loads(arguments)`** | OpenAI ships arguments as a string for transport reasons — you have to parse them. Gemini's `args` are already a dict. |
| **`role: "tool"` reply** | Function results go back as a message with `role="tool"`, `tool_call_id=<the id from the model>`, and `content=<JSON string of the result>`. Multiple tools in one round → multiple `role: "tool"` messages, each with its own id. |
| **Persisting the assistant turn** | OpenAI requires the assistant message (with its `tool_calls`) to be present in the next call's `messages` list. Append it before the tool messages. |
| **Two-round flow (same as Gemini)** | Round 1 returns `tool_calls`. Round 2 is the same `messages` plus the assistant turn plus the tool responses; the model produces the final user-facing text. |

## HOW (code walkthrough)

```python
def get_weather(city: str) -> dict:
    fake = {
        "mumbai":    {"temp_c": 31, "conditions": "humid, partly cloudy"},
        "bangalore": {"temp_c": 24, "conditions": "light rain"},
        "delhi":     {"temp_c": 36, "conditions": "hot and dry"},
    }
    return fake.get(city.lower(), {"error": f"No data for {city}"})
```

The same Python function as in the Gemini script. Tool calling is
provider-portable at this level — only the wrapping changes.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for an Indian city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Name of the city."},
            },
            "required": ["city"],
        },
    },
}]
```

Plain dicts, plain JSON Schema. No SDK-specific classes. This is also why
the OpenAI tool spec ports cleanly to most third-party providers and to
`response_format` schemas elsewhere.

```python
messages = [
    {"role": "user", "content": "What's the weather in Bangalore right now?"},
]

r1 = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
assistant_msg = r1.choices[0].message
messages.append(assistant_msg)               # critical: keep the tool_calls turn
```

Round 1: prompt + tool list goes in. The assistant message is appended
*before* we add the tool result — OpenAI rejects the next call if the
`tool_calls` turn is missing.

```python
if assistant_msg.tool_calls:
    for call in assistant_msg.tool_calls:
        args = json.loads(call.function.arguments)   # arguments is a string
        result = get_weather(**args) if call.function.name == "get_weather" else {"error": "unknown"}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })

    r2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    print(r2.choices[0].message.content)
```

For each requested tool call, run the function and append a
`role: "tool"` message with the matching `tool_call_id`. Round 2 with the
augmented messages produces the final answer.

### Side-by-side delta with Gemini

| | Gemini | OpenAI |
|---|---|---|
| Tool wrapper | `types.Tool(function_declarations=[FunctionDeclaration(...)])` | plain dict `{"type": "function", "function": {...}}` |
| Parameter schema | `types.Schema(type=Type.OBJECT, properties={...})` | plain JSON Schema dict |
| Where to read the call | `response.candidates[0].content.parts[0].function_call` | `response.choices[0].message.tool_calls[0].function` |
| `args` shape | dict already | JSON string (`json.loads` it) |
| Sending the result | `Part.from_function_response(name, response)` inside a `Content(role="user")` | message with `role="tool"`, `tool_call_id=<id>`, `content=<json string>` |
| Must replay assistant turn? | Yes (whole `response.candidates[0].content`) | Yes (`assistant_msg` with its `tool_calls`) |

**Run it:**

```bash
python 05_openai/02_tool_calling.py
```

Open `02_gemini/07_tool_calling.py` in another window. Read both flows in
parallel. You'll feel the muscle memory kick in by the second pass — the
*flow* is the same, the wrappers are not. That is the whole "delta view"
the workshop is teaching.
