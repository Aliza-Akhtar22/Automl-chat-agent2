from __future__ import annotations
import json
from src.agents.llm_client import get_client

def generate_code(client, model: str, codegen_prompt: str, plan: dict, df_summary: dict) -> str:
    user_msg = json.dumps(
        {"plan": plan, "data_summary": df_summary},
        indent=2,
        default=str,
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": codegen_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content
