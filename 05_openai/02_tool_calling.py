"""
05_openai/02_tool_calling.py
-----------------------------------------------------------------
Same weather example as 02_gemini/07_tool_calling.py, in OpenAI's syntax.

Differences worth pointing out in class:

    GEMINI                              OPENAI
    types.Tool(                         tools=[{
      function_declarations=[             "type": "function",
        types.FunctionDeclaration(...)    "function": {
      ]                                     "name": ..., "description": ...,
    )                                       "parameters": <JSON schema>
                                          }
                                        }]

    response.candidates[0]              response.choices[0]
      .content.parts[0]                   .message.tool_calls[0]
      .function_call                      .function

    Part.from_function_response(...)    {"role": "tool",
                                         "tool_call_id": ...,
                                         "content": ...}

The flow is identical: model proposes call, you execute, send result back,
model finalizes the answer.
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_weather(city: str) -> dict:
    fake = {
        "mumbai":    {"temp_c": 31, "conditions": "humid, partly cloudy"},
        "bangalore": {"temp_c": 24, "conditions": "light rain"},
        "delhi":     {"temp_c": 36, "conditions": "hot and dry"},
    }
    return fake.get(city.lower(), {"error": f"No data for {city}"})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for an Indian city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. 'Mumbai'.",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather in Bangalore right now? Should I carry an umbrella?",
    },
]

# ---- Round 1: model decides to call the tool --------------------------
r1 = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
)

assistant_msg = r1.choices[0].message
messages.append(assistant_msg)   # OpenAI requires the assistant turn in history

if assistant_msg.tool_calls:
    for call in assistant_msg.tool_calls:
        args = json.loads(call.function.arguments)
        print(f"Model wants to call : {call.function.name}({args})")

        if call.function.name == "get_weather":
            result = get_weather(**args)
        else:
            result = {"error": "unknown function"}
        print(f"Function result    : {result}")

        # Append the tool's result with role="tool" and the matching tool_call_id
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })

    # ---- Round 2: model writes the final user-facing answer -----------
    r2 = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools,
    )
    print("\nFinal answer:")
    print(r2.choices[0].message.content)
else:
    print("Model answered without a tool call:")
    print(assistant_msg.content)
