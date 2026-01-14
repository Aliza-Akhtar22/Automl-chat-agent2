from __future__ import annotations
import json

def explain_retry_fix(
    client,
    model: str,
    prompt: str,
    plan: dict,
    df_summary: dict,
    prev_code: str,
    error_text: str,
    corrected_code: str,
) -> str:
    user_msg = json.dumps(
        {
            "plan": plan,
            "data_summary": df_summary,
            "previous_code": prev_code,
            "error": error_text,
            "corrected_code": corrected_code,
        },
        indent=2,
        default=str,
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
