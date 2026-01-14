# src/core/data_summary.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def _is_datetime_candidate(col: str) -> bool:
    c = col.lower()
    keywords = ["date", "time", "timestamp", "datetime", "dt", "ds"]
    return any(k in c for k in keywords)


def _infer_freq_from_seconds(median_seconds: float) -> str:
    """
    Map median time delta (seconds) to a Prophet-friendly freq string.
    """
    if median_seconds <= 90:      # ~1 minute
        return "T"
    if median_seconds <= 60 * 90: # ~1 hour-ish
        return "H"
    if median_seconds <= 60 * 60 * 36:  # ~1 day-ish
        return "D"
    if median_seconds <= 60 * 60 * 24 * 10:  # ~weekly-ish
        return "W"
    return "M"


def _try_parse_datetime_series(s: pd.Series) -> Optional[pd.Series]:
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        if dt.notna().mean() < 0.7:
            return None
        return dt
    except Exception:
        return None


def _best_datetime_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Returns (best_col, inferred_freq, median_delta_seconds).
    """
    candidates = [c for c in df.columns if _is_datetime_candidate(c)]
    # If no keyword candidates, try object columns as fallback
    if not candidates:
        candidates = [c for c in df.columns if df[c].dtype == "object"]

    best = None
    best_parse_rate = 0.0
    best_freq = None
    best_median_sec = None

    for c in candidates:
        dt = _try_parse_datetime_series(df[c])
        if dt is None:
            continue

        parse_rate = dt.notna().mean()
        if parse_rate > best_parse_rate:
            # infer delta on sorted unique timestamps
            dts = dt.dropna().sort_values()
            if len(dts) >= 5:
                diffs = dts.diff().dropna().dt.total_seconds()
                if len(diffs) > 0:
                    med = float(diffs.median())
                    freq = _infer_freq_from_seconds(med)
                else:
                    med, freq = None, None
            else:
                med, freq = None, None

            best = c
            best_parse_rate = parse_rate
            best_freq = freq
            best_median_sec = med

    return best, best_freq, best_median_sec


def summarize_dataframe(df: pd.DataFrame, max_rows: int = 8) -> dict:
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    missing = df.isna().sum().to_dict()
    sample = df.head(max_rows).to_dict(orient="records")
    nunique = df.nunique(dropna=True).to_dict()

    # Numeric columns (exclude bools if you want)
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    # Datetime inference
    dt_candidates = [c for c in df.columns if _is_datetime_candidate(c)]
    best_dt, inferred_freq, median_delta_seconds = _best_datetime_column(df)

    # Basic numeric stats (useful for "meaningful y" selection)
    num_stats: Dict[str, Dict[str, Any]] = {}
    for c in numeric_cols[:50]:
        s = pd.to_numeric(df[c], errors="coerce")
        num_stats[c] = {
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "missing_frac": float(s.isna().mean()),
            "nunique": int(s.nunique(dropna=True)),
        }

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_by_column": missing,
        "nunique_by_column": {k: int(v) for k, v in nunique.items()},
        "sample_rows": sample,
        "numeric_columns": numeric_cols,
        "numeric_stats": num_stats,
        "datetime_candidates": dt_candidates,
        "best_datetime_col": best_dt,
        "inferred_freq": inferred_freq,
        "median_delta_seconds": median_delta_seconds,
    }
