from __future__ import annotations
import json
from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents.llm_client import chat_json

class ProphetPlan(BaseModel):
    task: str = Field(default="forecast_prophet")
    ds_col: str
    y_col: str
    regressors: List[str] = Field(default_factory=list)
    forecast_periods: int = 30
    freq: str = "D"
    notes: Optional[str] = None

def build_plan(client, model: str, planner_prompt: str, df_summary: dict, user_context: dict) -> ProphetPlan:
    user_msg = json.dumps(
        {"data_summary": df_summary, "user_context": user_context},
        indent=2,
        default=str,
    )

    raw = chat_json(
        client=client,
        model=model,
        system=planner_prompt,
        user=user_msg,
    )

    plan_dict = json.loads(raw)

    # Apply overrides if present (user UI wins)
    if user_context.get("ds_override"):
        plan_dict["ds_col"] = user_context["ds_override"]
    if user_context.get("y_override"):
        plan_dict["y_col"] = user_context["y_override"]
    if user_context.get("regressors_override") is not None:
        # keep as user-selected if provided (even empty)
        plan_dict["regressors"] = list(user_context["regressors_override"])
    if user_context.get("forecast_periods_override"):
        plan_dict["forecast_periods"] = int(user_context["forecast_periods_override"])
    if user_context.get("freq_override"):
        plan_dict["freq"] = user_context["freq_override"]

    return ProphetPlan(**plan_dict)
