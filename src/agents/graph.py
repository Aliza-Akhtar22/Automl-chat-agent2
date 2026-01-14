from __future__ import annotations

from typing import TypedDict, Optional, Any, Dict
from pathlib import Path

from langgraph.graph import StateGraph, END

from src.core.data_summary import summarize_dataframe
from src.core.execution import execute_generated_code
from src.agents.llm_client import get_client
from src.agents.planner import build_plan
from src.agents.codegen import generate_code
from src.agents.retry import regenerate_code_with_error
from src.agents.retry_explainer import explain_retry_fix


# --- Prompt loading (robust for Streamlit Cloud / varying CWD) ---
_BASE_DIR = Path(__file__).resolve().parents[1]  # .../src
_PROMPTS_DIR = _BASE_DIR / "prompts"

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

PLANNER_PROMPT = _read_text(_PROMPTS_DIR / "planner_prompt.txt")
CODEGEN_PROMPT = _read_text(_PROMPTS_DIR / "codegen_prophet_prompt.txt")
RETRY_PROMPT = _read_text(_PROMPTS_DIR / "retry_prompt.txt")
RETRY_EXPLAINER_PROMPT = _read_text(_PROMPTS_DIR / "retry_explainer_prompt.txt")

class AgentState(TypedDict, total=False):
    df: Any
    df_summary: dict
    plan: dict
    code: str
    execution: dict | None
    error_context: Optional[str]
    retries_left: int
    execute: bool
    api_key: str
    model_name: str
    user_context: dict
    force_retry: bool
    fix_explanation: Optional[str]


def node_summarize(state: AgentState) -> AgentState:
    return {"df_summary": summarize_dataframe(state["df"])}


def node_plan(state: AgentState) -> AgentState:
    client = get_client(state["api_key"])
    plan_obj = build_plan(
        client=client,
        model=state["model_name"],
        planner_prompt=PLANNER_PROMPT,
        df_summary=state["df_summary"],
        user_context=state["user_context"],
        df=state["df"],  # NEW: needed for deterministic regressor selection
    )
    return {"plan": plan_obj.model_dump()}


def node_codegen(state: AgentState) -> AgentState:
    client = get_client(state["api_key"])
    code = generate_code(
        client=client,
        model=state["model_name"],
        codegen_prompt=CODEGEN_PROMPT,
        plan=state["plan"],
        df_summary=state["df_summary"],
    )
    return {"code": code}


def node_execute(state: AgentState) -> AgentState:
    # For "copy-paste into VS Code" flow, you may set execute=False in run_agent
    if not state.get("execute", True):
        return {"execution": None}
    exec_res = execute_generated_code(state["code"], state["df"])
    return {"execution": exec_res}


def node_retry_codegen(state: AgentState) -> AgentState:
    client = get_client(state["api_key"])
    error_text = state.get("error_context") or (state.get("execution") or {}).get("stderr", "")

    corrected_code = regenerate_code_with_error(
        client=client,
        model=state["model_name"],
        retry_prompt=RETRY_PROMPT,
        plan=state["plan"],
        df_summary=state["df_summary"],
        prev_code=state["code"],
        error_text=error_text,
    )

    # NEW: explain what changed
    fix_expl = explain_retry_fix(
        client=client,
        model=state["model_name"],
        prompt=RETRY_EXPLAINER_PROMPT,
        plan=state["plan"],
        df_summary=state["df_summary"],
        prev_code=state["code"],
        error_text=error_text,
        corrected_code=corrected_code,
    )

    return {"code": corrected_code, "fix_explanation": fix_expl}


def should_retry(state: AgentState) -> str:
    """
    Two modes:
    1) execute=True  -> retry only if execution failed and retries_left > 0
    2) execute=False -> retry only if user forced retry (paste error) and retries_left > 0
    """
    retries_left = int(state.get("retries_left", 0))

    # User-pasted error regeneration flow (no execution inside app)
    if not state.get("execute", True):
        if state.get("force_retry") and retries_left > 0:
            return "retry"
        return END

    # Normal in-app execution flow
    exec_res = state.get("execution")
    if not exec_res:
        return END
    if exec_res.get("ok"):
        return END
    if retries_left <= 0:
        return END
    return "retry"


def decrement_retries(state: AgentState) -> AgentState:
    # Make force_retry one-shot; prevents looping when retries are exhausted.
    return {
        "retries_left": max(0, int(state.get("retries_left", 0)) - 1),
        "force_retry": False,
    }


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("summarize", node_summarize)
    g.add_node("plan", node_plan)
    g.add_node("codegen", node_codegen)
    g.add_node("execute", node_execute)
    g.add_node("retry_codegen", node_retry_codegen)
    g.add_node("decrement", decrement_retries)

    g.set_entry_point("summarize")
    g.add_edge("summarize", "plan")
    g.add_edge("plan", "codegen")
    g.add_edge("codegen", "execute")

    g.add_conditional_edges(
        "execute",
        should_retry,
        {
            "retry": "decrement",
            END: END,
        },
    )
    g.add_edge("decrement", "retry_codegen")
    g.add_edge("retry_codegen", "execute")

    return g.compile()


_GRAPH = build_graph()


def run_agent(
    df,
    api_key: str,
    model_name: str,
    user_context: dict,
    execute: bool = True,
    max_retries: int = 1,
    error_context: str | None = None,
    force_retry: bool = False,
) -> Dict[str, Any]:
    init_state: AgentState = {
        "df": df,
        "api_key": api_key,
        "model_name": model_name,
        "user_context": user_context,
        "execute": execute,
        "retries_left": int(max_retries),
        "error_context": error_context,
        "force_retry": force_retry,
    }

    out = _GRAPH.invoke(init_state)

    return {
        "df_summary": out.get("df_summary"),
        "plan": out.get("plan"),
        "code": out.get("code"),
        "execution": out.get("execution"),
        "fix_explanation": out.get("fix_explanation"),
    }
