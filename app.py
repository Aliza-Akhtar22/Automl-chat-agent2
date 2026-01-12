import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agents.graph import run_agent

load_dotenv()

st.set_page_config(page_title="Prophet Coding Agent", layout="wide")

def get_api_key() -> str | None:
    # Streamlit Cloud: use secrets; Local: use env/.env
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")

st.title("Prophet Coding Agent (Plan → Code → Execute → Retry)")

api_key = get_api_key()
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

model_name = st.sidebar.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

st.sidebar.markdown("---")
execute_in_app = st.sidebar.checkbox("Execute generated code in-app", value=True)
max_retries = st.sidebar.slider("Max retries", 0, 3, 1)

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
    freq_override = st.selectbox("Frequency", ["(auto)", "D", "W", "M", "H"])

user_context = {
    "ds_override": None if ds_override == "(auto)" else ds_override,
    "y_override": None if y_override == "(auto)" else y_override,
    "regressors_override": reg_override,
    "forecast_periods_override": int(periods_override),
    "freq_override": None if freq_override == "(auto)" else freq_override,
}

st.markdown("---")
run_btn = st.button("Generate plan + code (and optionally execute)", type="primary")

if run_btn:
    with st.spinner("Running agent..."):
        result = run_agent(
            df=df,
            api_key=api_key,
            model_name=model_name,
            user_context=user_context,
            execute=execute_in_app,
            max_retries=max_retries,
        )

    st.subheader("Plan")
    st.code(json.dumps(result["plan"], indent=2), language="json")

    st.subheader("Generated Code (copy-paste into VS Code)")
    st.code(result["code"], language="python")

    if result.get("execution"):
        st.subheader("Execution Result (in-app)")
        exec_res = result["execution"]
        if exec_res["ok"]:
            st.success("Execution succeeded.")
            st.text_area("Stdout", exec_res["stdout"], height=150)
            if exec_res["result_preview"]:
                st.markdown("**Result preview**")
                st.code(exec_res["result_preview"], language="json")
        else:
            st.error("Execution failed.")
            st.text_area("Stdout", exec_res["stdout"], height=120)
            st.text_area("Stderr", exec_res["stderr"], height=200)

            st.markdown("---")
            st.subheader("Retry")
            pasted_err = st.text_area("Paste error (or edit) and click Retry", value=exec_res["stderr"], height=180)
            if st.button("Retry with error context"):
                with st.spinner("Regenerating code with error context..."):
                    retry_result = run_agent(
                        df=df,
                        api_key=api_key,
                        model_name=model_name,
                        user_context=user_context,
                        execute=execute_in_app,
                        max_retries=max_retries,
                        error_context=pasted_err,
                        force_retry=True,
                    )
                st.subheader("Updated Code")
                st.code(retry_result["code"], language="python")
                if retry_result.get("execution"):
                    if retry_result["execution"]["ok"]:
                        st.success("Retry succeeded.")
                    else:
                        st.error("Retry failed again.")
                        st.text_area("Retry stderr", retry_result["execution"]["stderr"], height=200)
