# `07_tool_calling.py` — Concept Guide

## WHY (purpose)

A pure language model has no live data. It does not know the current time,
today's weather, your inventory levels, or what's in your CRM — its
training data is frozen. Tool calling closes that gap by letting the model
*ask your code* to fetch information, then continue its answer with the
real result baked in.

The point worth burning into the lesson is: **the model never executes
anything.** It tells you which function to call and with what arguments;
your application runs it; you send the result back. This is what keeps
tool calling safe — every action is gated by your code.

## WHAT (technical concepts)

| Concept | What it means |
|---|---|
| **Tool calling / function calling** | The protocol where the model emits a structured "please call `f(x)`" instead of plain text. Used to ground answers on live data, trigger workflows, query databases, etc. |
| **Function declaration** | A description of a function in a form the model can read: name, human description, parameter schema. The description is what the model uses to decide *when* to call it — write it as if for a human. |
| **`types.Tool` / `types.FunctionDeclaration`** | The Gemini wrappers for declaring tools. A `Tool` holds one or more `FunctionDeclaration`s; a `FunctionDeclaration` has a name, description, and a `types.Schema` for parameters. |
| **`types.Schema`** | Gemini's JSON-schema-shaped object. `type=Type.OBJECT` with named `properties` and a `required` list mirrors a typed function signature. |
| **`function_call` part** | When the model decides to use a tool, the response part is a `function_call` with `name` and `args` (a dict). No `text` is produced for that part. |
| **`Part.from_function_response`** | The way you send the function's *result* back. The model picks up where it left off and writes the final user-facing answer. |
| **The two-round pattern** | Round 1: prompt + tools → model returns function_call. You execute. Round 2: prompt + model's function_call + your function_response → model returns final text. |
| **Why it's "non-agentic" here** | The script handles exactly one tool call. Agentic systems loop — they let the model call multiple tools across multiple turns, often with planning. The mechanics are the same; agentic frameworks just wrap them in a loop. |

## HOW (code walkthrough)

```python
def get_weather(city: str) -> dict:
    fake_db = {
        "mumbai":    {"temp_c": 31, "conditions": "humid, partly cloudy"},
        "bangalore": {"temp_c": 24, "conditions": "light rain"},
        "delhi":     {"temp_c": 36, "conditions": "hot and dry"},
    }
    return fake_db.get(city.lower(), {"error": f"No data for {city}"})
```

A regular Python function. In production this would call a real weather
API; the contract is the same.

```python
weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for an Indian city.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="Name of the city, e.g. 'Mumbai'.",
                    ),
                },
                required=["city"],
            ),
        )
    ]
)
```

The `description` and parameter `description` strings are not decoration —
they are what the model reads when deciding whether to call the tool.
Vague descriptions lead to bad tool selection.

```python
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather in Bangalore right now? Should I carry an umbrella?",
    config=types.GenerateContentConfig(tools=[weather_tool]),
)
part = response.candidates[0].content.parts[0]
```

Round 1: the model chooses what to do. Inspect the part — it's either
`function_call` (model wants help) or text (model decided it can answer
without help).

```python
if part.function_call:
    fc = part.function_call
    result = get_weather(**fc.args)        # YOU run it

    follow_up = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user",  parts=[types.Part(text=user_prompt)]),
            response.candidates[0].content,                       # the function_call
            types.Content(role="user", parts=[
                types.Part.from_function_response(name=fc.name, response=result),
            ]),
        ],
        config=types.GenerateContentConfig(tools=[weather_tool]),
    )
    print(follow_up.text)
```

Round 2: replay the original prompt, the model's function_call turn, and
the result of running the function. The model now has live data and
finishes the answer.

**Run it:**

```bash
python 02_gemini/07_tool_calling.py
```

Try changing the city to one that's not in `fake_db`. The function
returns `{"error": ...}`; observe how the model surfaces that error
gracefully in its final answer rather than crashing the loop. That
graceful-degradation behaviour is half the reason tool calling beats
hard-coded if/else routing.
