import os

import requests
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"


def post_chat(messages):
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "reasoning": {"enabled": True},
        },
        timeout=120,
    )

    print("status_code:", response.status_code)
    if not response.ok:
        print("error_body:", response.text)
        response.raise_for_status()

    data = response.json()
    if "choices" not in data:
        raise KeyError(f"Missing 'choices' in response: {data}")

    return data["choices"][0]["message"]


initial_messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
]

first_message = post_chat(initial_messages)
print("first_content:", first_message.get("content"))

followup_messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": first_message.get("content"),
        "reasoning_details": first_message.get("reasoning_details"),
    },
    {"role": "user", "content": "Are you sure? Think carefully."},
]

second_message = post_chat(followup_messages)
print("second_content:", second_message.get("content"))
