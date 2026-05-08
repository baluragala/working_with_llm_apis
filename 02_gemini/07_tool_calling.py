"""
02_gemini/07_tool_calling.py
-----------------------------------------------------------------
Tool calling = letting the model invoke functions you defined.

Flow:
    1. You declare functions (name, description, parameters).
    2. You send a prompt + the tools list.
    3. The model EITHER replies in plain text OR returns a function_call
       saying "please run get_weather(city='Mumbai')".
    4. YOU run the function locally with the args it gave you.
    5. You send the function's result back as a tool response.
    6. The model finishes its answer, now grounded on real data.

The model never executes anything. It just asks. You decide.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ---- 1. The "real" function. Could call a weather API; we mock it. ----
def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    fake_db = {
        "mumbai":    {"temp_c": 31, "conditions": "humid, partly cloudy"},
        "bangalore": {"temp_c": 24, "conditions": "light rain"},
        "delhi":     {"temp_c": 36, "conditions": "hot and dry"},
    }
    return fake_db.get(city.lower(), {"error": f"No data for {city}"})


# ---- 2. Declare the tool to Gemini ------------------------------------
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

# ---- 3. Ask the model. It will choose to call the tool. ---------------
user_prompt = "What's the weather in Bangalore right now? Should I carry an umbrella?"

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_prompt,
    config=types.GenerateContentConfig(tools=[weather_tool]),
)

# ---- 4. Inspect what the model returned -------------------------------
part = response.candidates[0].content.parts[0]

if part.function_call:
    fc = part.function_call
    print(f"Model wants to call : {fc.name}({dict(fc.args)})")

    # ---- 5. Run the function on our side ------------------------------
    if fc.name == "get_weather":
        result = get_weather(**fc.args)
        print(f"Function result    : {result}")
    else:
        result = {"error": "unknown function"}

    # ---- 6. Send the result back so the model can finish --------------
    follow_up = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user",  parts=[types.Part(text=user_prompt)]),
            response.candidates[0].content,                # the function call
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=fc.name, response=result,
                )],
            ),
        ],
        config=types.GenerateContentConfig(tools=[weather_tool]),
    )
    print()
    print("Final answer:")
    print(follow_up.text)
else:
    print("Model answered without a tool call:")
    print(response.text)
