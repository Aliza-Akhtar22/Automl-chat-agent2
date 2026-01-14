import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agents.graph import run_agent
from src.agents.llm_client import get_client

# NEW: simple deterministic requirements for Prophet runs
def prophet_requirements_text() -> str:
    lines = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "prophet>=1.1.5",
        "cmdstanpy>=1.2.0",
    ]
    return "\n".join(lines) + "\n"


# NEW: natural-language explainer prompt (kept inline for copy-paste simplicity)
EXPLAINER_SYSTEM_PROMPT = """You are a helpful data scientist.

Given a dataset summary and a Prophet plan, explain in natural language:
- which column should be treated as ds and why
- which column should be treated as y and why
- which columns (if any) are regressors and why
- what frequency and forecast horizon you will use

Keep it short (4-8 lines). Do not output JSON. Do not output code.
"""


def explain_columns(client, model: str, plan: dict, df_summary: dict) -> str:
    user_msg = json.dumps({"plan": plan, "data_summary": df_summary}, indent=2, default=str)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


load_dotenv()

st.set_page_config(page_title="Prophet Coding Agent", layout="wide")


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_model_name() -> str:
    try:
        return st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    except Exception:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


st.title("Prophet Coding Agent (Plan → Code → Retry)")

api_key = get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

# Keep UI simple: model selector only
model_name = st.sidebar.text_input("OpenAI model", value=get_model_name())

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Data Preview")
st.dataframe(df.head(50), use_container_width=True)

with st.expander("Optional overrides (recommended if agent guesses wrong)", expanded=False):
    cols = list(df.columns)
    ds_override = st.selectbox("Date column (ds)", ["(auto)"] + cols)
    y_override = st.selectbox("Target column (y)", ["(auto)"] + cols)
    reg_override = st.multiselect("Regressors (optional)", cols, default=[])
    periods_override = st.number_input("Forecast periods", min_value=1, max_value=5000, value=30)
    freq_override = st.selectbox("Frequency", ["(auto)", "T", "H", "D", "W", "M"])

user_context = {
    "ds_override": None if ds_override == "(auto)" else ds_override,
    "y_override": None if y_override == "(auto)" else y_override,
    "regressors_override": reg_override,
    "forecast_periods_override": int(periods_override),
    "freq_override": None if freq_override == "(auto)" else freq_override,
}

# Persist last outputs for regeneration workflow
if "last_plan" not in st.session_state:
    st.session_state["last_plan"] = None
if "last_code" not in st.session_state:
    st.session_state["last_code"] = None
if "last_df_summary" not in st.session_state:
    st.session_state["last_df_summary"] = None

st.markdown("---")

run_btn = st.button("Generate plan + code", type="primary")

if run_btn:
    with st.spinner("Running agent..."):
        # IMPORTANT: Prophet code should NOT be executed inside Streamlit (sandbox/import restrictions).
        # This app is "generate → copy to VS Code → paste error → regenerate".
        result = run_agent(
            df=df,
            api_key=api_key,
            model_name=model_name,
            user_context=user_context,
            execute=False,     # ALWAYS False
            max_retries=1,     # internal; not shown in UI
        )

    st.session_state["last_plan"] = result.get("plan")
    st.session_state["last_code"] = result.get("code")
    st.session_state["last_df_summary"] = result.get("df_summary")

    # Natural-language explanation
    try:
        client = get_client(api_key)
        explanation = explain_columns(
            client=client,
            model=model_name,
            plan=result["plan"],
            df_summary=result["df_summary"],
        )
        st.subheader("What I inferred from your data")
        st.write(explanation)
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")

    # Plan JSON
    st.subheader("Plan")
    st.code(json.dumps(result["plan"], indent=2), language="json")

    # requirements.txt before code
    st.subheader("requirements.txt (install these before running)")
    st.code(prophet_requirements_text(), language="text")
    st.caption("Run: pip install -r requirements.txt")

    # Code
    st.subheader("Generated Code (copy-paste into VS Code)")
    st.code(result["code"], language="python")

st.markdown("---")

# Error → regenerate workflow (primary UX)
st.subheader("Fix an error (paste your VS Code error and regenerate code)")
pasted_err = st.text_area(
    "Paste error here",
    height=220,
    placeholder="Paste the full traceback / error message here (from VS Code / terminal)...",
)

regen_btn = st.button("Regenerate code from error")

if regen_btn:
    if (
        not st.session_state.get("last_code")
        or not st.session_state.get("last_plan")
        or not st.session_state.get("last_df_summary")
    ):
        st.error("First click 'Generate plan + code' so I have the plan and original code to correct.")
    elif not pasted_err.strip():
        st.error("Please paste an error message/traceback first.")
    else:
        with st.spinner("Regenerating code using your error..."):
            retry_result = run_agent(
                df=df,
                api_key=api_key,
                model_name=model_name,
                user_context=user_context,
                execute=False,
                max_retries=1,        # internal
                error_context=pasted_err,
                force_retry=True,
            )

        st.session_state["last_code"] = retry_result.get("code")

        if retry_result.get("fix_explanation"):
            st.subheader("What was fixed")
            st.write(retry_result["fix_explanation"])

        st.subheader("Regenerated Code")
        st.code(retry_result["code"], language="python")
        st.caption("Copy-paste the regenerated code into VS Code and run again.")
