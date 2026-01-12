from __future__ import annotations
import pandas as pd

def summarize_dataframe(df: pd.DataFrame, max_rows: int = 8) -> dict:
    # Keep summary compact for prompts
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    missing = df.isna().sum().to_dict()
    sample = df.head(max_rows).to_dict(orient="records")
    nunique = df.nunique(dropna=True).to_dict()

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_by_column": missing,
        "nunique_by_column": {k: int(v) for k, v in nunique.items()},
        "sample_rows": sample,
    }
