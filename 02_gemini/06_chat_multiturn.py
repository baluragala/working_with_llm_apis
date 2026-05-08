"""
02_gemini/06_chat_multiturn.py
-----------------------------------------------------------------
LLMs are stateless. To have a "conversation", you re-send the full history
on every turn. Two ways to do this with Gemini:

    APPROACH 1 - Manual:  Build a list of Content objects yourself, append after
                          each turn, and pass it as `contents`. Maximum control.

    APPROACH 2 - Chat:    client.chats.create(...) gives you a session object
                          that tracks history for you. Less boilerplate.

Roles in Gemini are slightly different from OpenAI:
    "user"   -> what the human said
    "model"  -> what Gemini said      (OpenAI calls this "assistant")
    System prompts are NOT a role. They go in `system_instruction` instead.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# =====================================================================
# APPROACH 1 — Manual history management
# =====================================================================
print("=" * 60)
print(" APPROACH 1: manual history list ")
print("=" * 60)

history = [
    types.Content(role="user", parts=[types.Part(text="My name is Asha. Remember it.")]),
]

# Turn 1
r1 = client.models.generate_content(model="gemini-2.5-flash", contents=history)
print(f"User : My name is Asha. Remember it.")
print(f"Model: {r1.text.strip()}")
history.append(types.Content(role="model", parts=[types.Part(text=r1.text)]))

# Turn 2 — should remember the name from turn 1
history.append(types.Content(role="user", parts=[types.Part(text="What is my name?")]))
r2 = client.models.generate_content(model="gemini-2.5-flash", contents=history)
print(f"User : What is my name?")
print(f"Model: {r2.text.strip()}")

# =====================================================================
# APPROACH 2 — chats module (history handled for you)
# =====================================================================
print()
print("=" * 60)
print(" APPROACH 2: chats module ")
print("=" * 60)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a concise assistant. Keep answers under 30 words.",
    ),
)

for user_msg in [
    "I'm planning a 3-day trip to Goa.",
    "Suggest one beach for day 1.",
    "And what about food there?",
]:
    reply = chat.send_message(user_msg)
    print(f"\nUser : {user_msg}")
    print(f"Model: {reply.text.strip()}")

# Chat object stores history; you can inspect it:
print("\n--- Stored history length:", len(chat.get_history()), "turns ---")
