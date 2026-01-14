from __future__ import annotations

def prophet_requirements() -> str:
    # Keep minimal and stable for running the generated script
    # You can pin versions later after testing on Streamlit Cloud.
    lines = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "prophet>=1.1.5",
        "cmdstanpy>=1.2.0",
    ]
    return "\n".join(lines) + "\n"
