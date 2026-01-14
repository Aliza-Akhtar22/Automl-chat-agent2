from __future__ import annotations

import io
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional

import pandas as pd


def _strip_markdown_fences(code: str) -> str:
    """
    If the model accidentally returns ```python ... ``` fences, strip them.
    Safe even if there are no fences.
    """
    s = code.strip()

    # Common patterns: ```python\n...\n``` or ```\n...\n```
    if s.startswith("```"):
        lines = s.splitlines()
        # Drop first line (``` or ```python)
        lines = lines[1:]
        # Drop last fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    return s


def execute_generated_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes model-generated Python code.

    Expected contract in generated code:
        def run(df: pd.DataFrame) -> dict:
            ...
            return {...}  # JSON-serializable preferred

    Returns:
        {
          "ok": bool,
          "stdout": str,
          "stderr": str,
          "result_preview": Optional[str],
        }
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    cleaned_code = _strip_markdown_fences(code)

    # Restrict builtins minimally (MVP). For production, sandboxing must be stronger.
    # NOTE: This is not a security sandbox. It only reduces accidental footguns.
    safe_builtins = {
        # basics
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
        # exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
    }

    g: Dict[str, Any] = {"__builtins__": safe_builtins}
    l: Dict[str, Any] = {}

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            compiled = compile(cleaned_code, "<generated_code>", "exec")
            exec(compiled, g, l)

            # Depending on how exec is called, function defs may land in globals or locals.
            fn = l.get("run") or g.get("run")
            if fn is None or not callable(fn):
                raise RuntimeError("Generated code must define a callable function: run(df) -> dict")

            # Run on a copy to avoid mutating the original df across retries.
            result = fn(df.copy())

        # Build a preview for UI logs
        result_preview: Optional[str]
        try:
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
            "stderr": stderr_buf.getvalue() + ("\n" if stderr_buf.getvalue() else "") + tb,
            "result_preview": None,
        }
