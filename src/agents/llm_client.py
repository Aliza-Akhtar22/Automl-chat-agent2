from __future__ import annotations
from openai import OpenAI

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def chat_json(client: OpenAI, model: str, system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content  # string JSON
