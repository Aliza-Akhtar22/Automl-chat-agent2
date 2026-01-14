from __future__ import annotations
import json

def explain_columns(client, model: str, explainer_prompt: str, plan: dict, df_summary: dict) -> str:
    user_msg = json.dumps({"plan": plan, "data_summary": df_summary}, indent=2, default=str)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": explainer_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()
