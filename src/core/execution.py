from __future__ import annotations
import io
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

def execute_generated_code(code: str, df: pd.DataFrame) -> dict:
    """
    Expects generated code to define:
        def run(df: pd.DataFrame) -> dict:
            ...
            return {"forecast_head": ..., ...}
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Restrict builtins minimally (MVP). For production, sandboxing should be stronger.
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "Exception": Exception,
        "ValueError": ValueError,
    }

    g = {"__builtins__": safe_builtins}
    l = {}

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            compiled = compile(code, "<generated_code>", "exec")
            exec(compiled, g, l)

            if "run" not in l:
                raise RuntimeError("Generated code must define a function: run(df) -> dict")

            result = l["run"](df.copy())

        result_preview = None
        try:
            # Attempt to JSON preview if possible
            result_preview = json.dumps(result, indent=2, default=str)[:3000]
        except Exception:
            result_preview = str(result)[:3000]

        return {
            "ok": True,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result_preview": result_preview,
        }

    except Exception:
        tb = traceback.format_exc()
        return {
            "ok": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue() + "\n" + tb,
            "result_preview": None,
        }
