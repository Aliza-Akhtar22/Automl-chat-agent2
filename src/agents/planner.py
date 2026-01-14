# src/agents/planner.py
from __future__ import annotations

import json
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.agents.llm_client import chat_json


class ProphetPlan(BaseModel):
    task: str = Field(default="forecast_prophet")
    ds_col: str
    y_col: str
    regressors: List[str] = Field(default_factory=list)
    forecast_periods: int = 30
    freq: str = "D"
    notes: Optional[str] = None


def _safe_datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)


def _pick_regressors(
    df: pd.DataFrame,
    ds_col: str,
    y_col: str,
    max_regressors: int = 6,
    corr_threshold: float = 0.12,
) -> List[str]:
    """
    Deterministic regressor selection:
    - numeric columns only
    - exclude ds/y
    - rank by abs(correlation with y)
    - enforce minimum correlation threshold
    """
    if y_col not in df.columns or ds_col not in df.columns:
        return []

    # Build numeric frame
    num_cols = [
        c for c in df.columns
        if c not in {ds_col, y_col}
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if not num_cols:
        return []

    y = pd.to_numeric(df[y_col], errors="coerce")
    # Require some signal
    if y.notna().sum() < 50:
        return []

    scored = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        common = pd.concat([y, x], axis=1).dropna()
        if len(common) < 100:
            continue
        corr = common.iloc[:, 0].corr(common.iloc[:, 1])
        if corr is None or pd.isna(corr):
            continue
        scored.append((c, float(abs(corr))))

    scored.sort(key=lambda t: t[1], reverse=True)
    picked = [c for c, s in scored if s >= corr_threshold][:max_regressors]
    return picked


def build_plan(
    client,
    model: str,
    planner_prompt: str,
    df_summary: dict,
    user_context: dict,
    df: pd.DataFrame,
) -> ProphetPlan:
    """
    Hybrid approach:
      1) Ask LLM for initial plan
      2) Apply deterministic fixes:
         - enforce ds using inferred best datetime
         - apply user overrides
         - infer freq from data when not overridden
         - add regressors when meaningful
    """
    user_msg = json.dumps(
        {"data_summary": df_summary, "user_context": user_context},
        indent=2,
        default=str,
    )

    raw = chat_json(client=client, model=model, system=planner_prompt, user=user_msg)
    plan_dict = json.loads(raw)

    # --- Enforce DS column deterministically when available ---
    inferred_ds = df_summary.get("best_datetime_col")
    if inferred_ds and inferred_ds in df.columns:
        plan_dict["ds_col"] = inferred_ds

    # --- Apply overrides (user UI wins) ---
    if user_context.get("ds_override"):
        plan_dict["ds_col"] = user_context["ds_override"]
    if user_context.get("y_override"):
        plan_dict["y_col"] = user_context["y_override"]

    # Forecast periods override
    if user_context.get("forecast_periods_override"):
        plan_dict["forecast_periods"] = int(user_context["forecast_periods_override"])

    # --- Frequency selection ---
    # If user did not override, use inferred freq from df_summary (T/H/D/W/M)
    if user_context.get("freq_override"):
        plan_dict["freq"] = user_context["freq_override"]
    else:
        inferred_freq = df_summary.get("inferred_freq")
        if inferred_freq:
            plan_dict["freq"] = inferred_freq
        else:
            # Safe fallback
            plan_dict["freq"] = plan_dict.get("freq") or "D"

    # --- Regressors selection ---
    # If user explicitly selected regressors in UI, keep them (even empty list)
    if user_context.get("regressors_override") is not None:
        plan_dict["regressors"] = list(user_context["regressors_override"])
    else:
        # If LLM chose regressors, keep them, but validate
        regs = plan_dict.get("regressors") or []
        regs = [r for r in regs if r in df.columns and r not in {plan_dict["ds_col"], plan_dict["y_col"]}]
        plan_dict["regressors"] = regs

        # If empty, try deterministic selection (this is what you want)
        if not plan_dict["regressors"]:
            ds_col = plan_dict["ds_col"]
            y_col = plan_dict["y_col"]
            if ds_col in df.columns and y_col in df.columns:
                picked = _pick_regressors(df=df, ds_col=ds_col, y_col=y_col)
                plan_dict["regressors"] = picked

    # Validate ds/y exist
    if plan_dict.get("ds_col") not in df.columns:
        raise ValueError(f"Planner picked ds_col={plan_dict.get('ds_col')} which is not in dataframe columns.")
    if plan_dict.get("y_col") not in df.columns:
        raise ValueError(f"Planner picked y_col={plan_dict.get('y_col')} which is not in dataframe columns.")

    # Notes (add deterministic explanation)
    notes_parts = []
    if df_summary.get("best_datetime_col"):
        notes_parts.append(f"ds inferred as '{plan_dict['ds_col']}'.")
    if df_summary.get("inferred_freq"):
        notes_parts.append(f"freq inferred as '{plan_dict['freq']}'.")
    if plan_dict.get("regressors"):
        notes_parts.append(f"regressors selected: {plan_dict['regressors']}.")
    else:
        notes_parts.append("no regressors selected (insufficient correlation/signal).")

    existing_notes = plan_dict.get("notes") or ""
    plan_dict["notes"] = (existing_notes + " " + " ".join(notes_parts)).strip()

    return ProphetPlan(**plan_dict)
