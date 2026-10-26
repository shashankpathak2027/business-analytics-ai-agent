"""
graph.py
--------
Core orchestration module for the Enterprise Business Analytics AI Agent.
This file is the brain of Analytica.
Everything intelligent happens here.
"""

import os 
import re 
import json 
import uuid
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field 
from langsmith import traceable
from langchain_experimental.utilities import PythonREPL 
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import InMemorySaver
from backend.config import llm_router, llm_reasoning, llm_coder, llm_report, CHARTS_DIR


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    csv_path: str
    user_query: str
    dataset_profile: Dict[str, Any]
    intent: str
    analysis_plan: str
    analysis_results: str
    chart_paths: List[Dict[str, str]] 
    insights: str
    final_report: str

MAX_CHARTS = 4 #can be changed in future if anyone needs extra graph
_ERROR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b\s*[:\(]")

# HELPERS FUNCTIONS (WE HAVE 3)------------
def _strip_code_fences(code: str) -> str:
    """Removes markdown code fences from LLM-generated code, if present."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()

def _execution_failed(output: str) -> bool:
    """Detects whether a PythonREPL output string is actually a captured
    exception rather than genuine printed analysis output."""
    if not output:
        return False
    stripped = output.strip()
    if not stripped:
        return False
    if "Traceback (most recent call last)" in stripped:
        return True
    first_line = stripped.splitlines()[0]
    return bool(_ERROR_PATTERN.match(first_line))


def _create_repl(csv_path: str) -> PythonREPL:
    """Creates a Python REPL with pandas/numpy/scipy/matplotlib preloaded
    and the uploaded CSV already loaded as `df`. Warnings are suppressed so
    non-fatal matplotlib/pandas warnings do not pollute captured output."""
    repl = PythonREPL()
    setup_code = (
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import scipy.stats as stats\n"
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        f"df = pd.read_csv(r'{csv_path}')\n"
    )
    repl.run(setup_code)
    return repl


# TOOLS(WE HAVE 3)-------
@tool
def profile_dataset(csv_path: str) -> str:
    """Reads the uploaded CSV file and returns a JSON profile containing
    shape, column names, data types, missing values, duplicate row count,
    summary statistics, and sample rows."""
    df = pd.read_csv(csv_path)
    profile = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "summary_statistics": json.loads(df.describe(include="all").fillna("").to_json()),
        "sample_rows": json.loads(df.head(5).to_json(orient="records")),
    }
    return json.dumps(profile, indent=2, default=str)


@tool
def run_analysis_code(csv_path: str, code: str) -> str:
    """Executes Python pandas/numpy/scipy code against the uploaded dataset
    (already loaded as `df`) and returns the printed output. The code must
    use print() statements to output every result that should be captured.
    If the code fails, the returned string contains the error text instead
    of raising - callers must check for failure themselves."""
    repl = _create_repl(csv_path)
    return str(repl.run(code))


@tool
def generate_chart(csv_path: str, code: str) -> str:
    """Executes matplotlib plotting code against the uploaded dataset
    (already loaded as `df`) and saves the resulting chart as a PNG file.
    The code should build a chart using `plt`. Returns the saved file path,
    or an empty string if the code failed to execute (callers should skip
    any chart with an empty return value rather than treating it as saved)."""
    repl = _create_repl(csv_path)
    result = str(repl.run(code))
    if _execution_failed(result):
        return ""
    filename = f"chart_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(CHARTS_DIR, filename)
    repl.run(f"plt.savefig(r'{filepath}', bbox_inches='tight', dpi=100)")
    repl.run("plt.close('all')")
    return filepath
 

# STRUCTURED OUTPUT SCHEMAS(4, OR YOU CAN SAY 3 MOSTLY)
class _RouterDecision(BaseModel):
    intent: Literal["new_analysis", "follow_up"] = Field(
        description=(
            "'follow_up' if the user's message can be answered using the "
            "analysis already completed (summarizing, explaining, "
            "clarifying, rephrasing, discussing an existing chart or "
            "finding). 'new_analysis' if it requires computing something "
            "new (a new metric, segment, forecast, or breakdown) that is "
            "not already covered."
        )
    )


class _FollowUpAnswer(BaseModel):
    reply: str = Field(
        description="Direct conversational answer to the user's follow-up "
        "question, using only the existing analysis as evidence."
    )
    updated_report: Optional[str] = Field(
        default=None,
        description="Only set this if the user explicitly asked to rewrite, "
        "regenerate, or otherwise revise the executive report itself: the "
        "complete revised report in Markdown, using only the existing "
        "analysis results as evidence. Leave null for any other follow-up.",
    )


class _ChartSpec(BaseModel):
    title: str = Field(description="Short, business-friendly chart title.")
    matplotlib_code: str = Field(
        description="Self-contained matplotlib code using `plt` and the "
        "preloaded dataframe `df`. Must call plt.figure() first. Must not "
        "call plt.show() or plt.savefig()."
    )
    explanation: str = Field(
        description="1-3 sentence business interpretation of what this "
        "chart shows and why it matters, shown directly under the chart."
    )


class _ChartPlan(BaseModel):
    charts: List[_ChartSpec] = Field(
        description=(
            f"At most {MAX_CHARTS} charts, ordered by business value "
            "(most valuable first). Propose fewer than the maximum if "
            "fewer are genuinely useful - never pad the list."
        )
    )


# AGENT NODES(WE HAVE 9 NODES AND 2 CONDITION FUNCTIONS FOR ROUTING)
@traceable(name="Planner Agent")
def planner_agent(state: AgentState) -> dict:
    """Reads the latest user message and stores it as the active query."""
    last_message = state["messages"][-1]
    user_query = last_message.content
    return {"user_query": user_query}


def route_after_planner(state: AgentState) -> str:
    """Routes to Dataset Understanding only if the dataset has not been
    profiled yet in this thread. Otherwise skips straight to intent
    routing."""
    if state.get("dataset_profile"):
        return "intent_router"
    return "dataset_understanding"


@traceable(name="Dataset Understanding Agent")
def dataset_understanding_agent(state: AgentState) -> dict:
    """Profiles the uploaded dataset and summarizes it for the user."""
    profile_json = profile_dataset.invoke({"csv_path": state["csv_path"]})
    profile = json.loads(profile_json)

    summary_prompt = (
        "You are a senior business data analyst reviewing a new dataset for "
        "the first time. Summarize the dataset profile below in 5-8 sentences "
        "for a business audience: mention row/column count, the key "
        "business-relevant columns you notice (e.g. dates, revenue, profit, "
        "customer, region, product), and any data quality issues such as "
        "missing values or duplicates. Do not invent information that is not "
        "present in the profile.\n\n"
        f"Dataset Profile:\n{profile_json}"
    )
    summary = llm_reasoning.invoke(summary_prompt).content

    return {
        "dataset_profile": profile,
        "messages": [AIMessage(content=f"Dataset Understanding:\n{summary}")],
    }


@traceable(name="Intent Router")
def intent_router(state: AgentState) -> dict:
    """Classifies whether the latest question needs a fresh data-analysis
    pass or can be answered directly from the analysis already in state.
    This is the fix for follow-up questions incorrectly re-running (and
    sometimes breaking) the full analysis pipeline."""
    if not state.get("analysis_results"):
        # Nothing has been analyzed yet in this thread - always a fresh run.
        return {"intent": "new_analysis"}

    prompt = (
        "Decide whether the user's latest message requires running NEW data "
        "analysis code, or whether it can be answered directly using the "
        "analysis already completed in this conversation.\n\n"
        f"User's latest message: {state['user_query']}\n\n"
        f"Existing analysis plan:\n{state.get('analysis_plan', '')}\n\n"
        f"Existing analysis results:\n{state.get('analysis_results', '')}\n\n"
        f"Existing insights:\n{state.get('insights', '')}"
    )
    decision = llm_router.with_structured_output(_RouterDecision).invoke(prompt)
    return {"intent": decision.intent}



def route_after_intent(state: AgentState) -> str:
    """Routes follow-up questions to the lightweight response agent, and
    everything else through the full analysis pipeline."""
    if state.get("intent") == "follow_up":
        return "follow_up_response"
    return "analysis_planning"



@traceable(name="Follow-Up Response Agent")
def follow_up_response_agent(state: AgentState) -> dict:
    """Answers a conversational follow-up question using only the analysis
    already produced in this thread - no new code execution, no new charts.
    This keeps follow-up questions fast, reliable, and grounded in existing
    evidence instead of re-running the full analysis pipeline."""
    prompt = (
        "You are a senior business analyst continuing a conversation with a "
        "client. Answer the client's latest message using ONLY the analysis "
        "already produced below - never invent new numbers or new findings. "
        "If the existing analysis genuinely does not contain what is needed "
        "to answer, say so honestly and suggest what new analysis would be "
        "needed instead.\n\n"
        f"Client's latest message: {state['user_query']}\n\n"
        f"Analysis plan:\n{state.get('analysis_plan', '')}\n\n"
        f"Analysis results:\n{state.get('analysis_results', '')}\n\n"
        f"Insights:\n{state.get('insights', '')}\n\n"
        f"Current executive report:\n{state.get('final_report', '')}"
    )
    answer = llm_reasoning.with_structured_output(_FollowUpAnswer).invoke(prompt)

    updates: Dict[str, Any] = {
        "insights": answer.reply,
        "messages": [AIMessage(content=answer.reply)],
    }
    if answer.updated_report:
        updates["final_report"] = answer.updated_report
    return updates


@traceable(name="Analysis Planning Agent")
def analysis_planning_agent(state: AgentState) -> dict:
    """Produces a rich, structured execution plan before analysis begins."""
    prompt = (
        "You are a senior business data analyst creating an execution plan. "
        "Think like an experienced consultant, not a generic assistant.\n\n"
        f"User question: {state['user_query']}\n\n"
        f"Dataset profile:\n{json.dumps(state['dataset_profile'], indent=2, default=str)}\n\n"
        "Using ONLY the columns that exist in the dataset profile above, "
        "produce a concrete execution plan with these labeled sections:\n\n"
        "ANALYSES NEEDED: list the specific descriptive and business analyses "
        "required to answer the question (e.g. group revenue by region, "
        "compare month-over-month trend, correlation between discount and "
        "profit).\n"
        "STATISTICAL TESTS: list any statistical tests or measures needed "
        "(e.g. correlation coefficients, t-test between two groups, "
        "confidence interval) — state 'None required' if not applicable.\n"
        "FORECASTING REQUIRED: state Yes or No, and justify briefly based on "
        "whether a usable date/time column exists and the question calls for "
        "it.\n"
        "SEGMENTATION REQUIRED: state Yes or No, and specify which dimension "
        "(customer, product, region, category) if Yes.\n"
        "ROOT CAUSE ANALYSIS REQUIRED: state Yes or No, and list candidate "
        "factors to investigate if Yes.\n"
        "CHARTS NEEDED: list the specific chart types that would best "
        "support this analysis, and the exact columns each chart should use.\n\n"
        "Do not assume columns that are not listed in the dataset profile. "
        "Be concrete and specific, not generic."
    )
    plan = llm_reasoning.invoke(prompt).content
    return {"analysis_plan": plan}


@traceable(name="Data Analysis Agent")
def data_analysis_agent(state: AgentState) -> dict:
    """Generates and executes thorough Python code to carry out the analysis
    plan. Retries once with a repair prompt if the first attempt fails, and
    never lets a raw Python exception reach the final report."""
    code_prompt = (
        "You are a senior data analyst. Write pandas/numpy/scipy code to "
        "fully execute the analysis plan below against a dataframe called "
        "`df` that is already loaded. Compute real numbers — do not just "
        "describe what could be done.\n\n"
        "Cover, where relevant to the plan and supported by the available "
        "columns:\n"
        "- Descriptive statistics (counts, means, medians, std, min/max) for "
        "key numeric columns\n"
        "- Missing values and duplicate counts\n"
        "- Outlier detection (e.g. IQR method) for key numeric columns\n"
        "- Correlation between relevant numeric columns\n"
        "- Group-by breakdowns requested in the plan (e.g. by region, "
        "category, product, customer)\n"
        "- Trend-over-time calculations if a usable date column exists\n"
        "- Statistical tests named in the plan (e.g. scipy.stats correlation "
        "p-values, t-tests between two groups) — only if the plan calls for "
        "them\n"
        "- Simple forecasting via numpy.polyfit trend extrapolation or "
        "pandas rolling means ONLY if the plan states forecasting is "
        "required — do not use external forecasting libraries, none are "
        "installed\n"
        "- Segmentation (e.g. top N and bottom N customers/products/regions) "
        "ONLY if the plan states segmentation is required\n\n"
        "Use print() with clear labels for every number you compute, so all "
        "results can be captured as text (e.g. print('Correlation revenue "
        "vs profit:', value)). Only output raw Python code, no markdown "
        "fences, no explanation.\n\n"
        f"Analysis plan:\n{state['analysis_plan']}\n\n"
        f"Dataset columns and types:\n"
        f"{json.dumps(state['dataset_profile'].get('dtypes', {}), indent=2)}"
    )
    code = _strip_code_fences(llm_coder.invoke(code_prompt).content)
    output = run_analysis_code.invoke({"csv_path": state["csv_path"], "code": code})

    if _execution_failed(output):
        repair_prompt = (
            "The following pandas/numpy/scipy code failed to execute "
            "against the dataframe `df`. Fix it and return the complete "
            "corrected code only — no markdown fences, no explanation.\n\n"
            f"Original code:\n{code}\n\n"
            f"Error:\n{output}"
        )
        code = _strip_code_fences(llm_coder.invoke(repair_prompt).content)
        output = run_analysis_code.invoke({"csv_path": state["csv_path"], "code": code})

    if _execution_failed(output):
        # Never let a raw Python exception reach insight/report generation.
        return {
            "analysis_results": (
                "Some parts of the requested analysis could not be computed "
                "successfully and were skipped. The findings below reflect "
                "only what could be reliably calculated from the dataset."
            )
        }

    return {"analysis_results": f"Code executed:\n{code}\n\nOutput:\n{output}"}


@traceable(name="Visualization Agent")
def visualization_agent(state: AgentState) -> dict:
    """Chooses at most MAX_CHARTS charts that provide the highest business
    value for this dataset and question, generates each one, and attaches a
    title and business explanation to every chart that renders successfully.
    Charts are REPLACED (not appended) so each new analysis produces its own
    fresh set instead of accumulating across every turn in the thread."""
    planning_prompt = (
        f"You are a senior data visualization expert. Choose AT MOST "
        f"{MAX_CHARTS} charts that provide the highest business value for "
        "the analysis below — do not pad the list; propose fewer if fewer "
        "are genuinely useful. Only propose a chart if the columns it needs "
        "actually exist in the dataset. Consider options such as: "
        "sales/revenue trend, profit trend, category performance, regional "
        "performance, correlation heatmap, customer segmentation, and "
        "top/bottom products — but only the ones that genuinely answer the "
        "analysis below, not every option on this list.\n\n"
        "Each chart's matplotlib_code must be self-contained: call "
        "plt.figure() first, use the preloaded dataframe `df`, and never "
        "call plt.show() or plt.savefig(). When labeling categorical x-axis "
        "ticks, call ax.set_xticks(...) before ax.set_xticklabels(...) to "
        "avoid tick-label warnings.\n\n"
        f"Dataset columns and types:\n"
        f"{json.dumps(state['dataset_profile'].get('dtypes', {}), indent=2)}\n\n"
        f"Analysis plan:\n{state['analysis_plan']}\n\n"
        f"Analysis results:\n{state['analysis_results']}"
    )
    plan = llm_coder.with_structured_output(_ChartPlan).invoke(planning_prompt)

    charts: List[Dict[str, str]] = []
    for spec in plan.charts[:MAX_CHARTS]:
        chart_path = generate_chart.invoke(
            {
                "csv_path": state["csv_path"],
                "code": _strip_code_fences(spec.matplotlib_code),
            }
        )
        if not chart_path:
            # Chart failed to render — skip it silently instead of saving and returning a blank image.
            continue
        charts.append(
            {"path": chart_path, "title": spec.title, "explanation": spec.explanation}
        )

    return {"chart_paths": charts}


@traceable(name="Insight Generation Agent")
def insight_generation_agent(state: AgentState) -> dict:
    """Derives evidence-backed, business-analyst-quality insights from the
    analysis results."""
    prompt = (
        "You are a senior business analyst. Based ONLY on the analysis "
        "results below, write clear, evidence-backed business insights that "
        "directly answer the user's question. Think in terms of business "
        "impact, not just numbers.\n\n"
        "Organize your answer under short thematic headings relevant to "
        "what was found (for example: Revenue & Profit, Product "
        "Performance, Regional Performance, Customers, Trends, Risks — "
        "include only the themes that are actually supported by the "
        "analysis results). For each point, reference the specific numbers "
        "from the analysis results that support it. If the evidence is "
        "insufficient to fully answer the question, explicitly say so "
        "rather than guessing. Do not invent numbers or facts that are not "
        "in the analysis results.\n\n"
        f"User question: {state['user_query']}\n\n"
        f"Analysis results:\n{state['analysis_results']}"
    )
    insights = llm_reasoning.invoke(prompt).content
    return {"insights": insights}


@traceable(name="Report Generation Agent")
def report_generation_agent(state: AgentState) -> dict:
    """Compiles a comprehensive, consulting-style executive business report
    that references the generated charts."""
    chart_paths = state.get("chart_paths") or []
    chart_list = (
        "\n".join(
            f"Chart {i + 1}: {c['title']} — {c['explanation']}"
            for i, c in enumerate(chart_paths)
        )
        or "No charts were generated for this analysis."
    )

    prompt = (
        "You are a senior management consultant preparing a professional "
        "business analytics report for executive stakeholders. Using ONLY "
        "the information below, write a comprehensive, evidence-based "
        "report in Markdown. Do not invent facts that are not supported by "
        "the analysis below — if a section does not apply (e.g. no "
        "forecasting was performed), briefly state that it is not "
        "applicable instead of fabricating content.\n\n"
        "Every claim must cite the specific number, percentage, or trend "
        "that supports it — never state a finding without the evidence "
        "behind it. Start the Executive Summary with one direct sentence "
        "that answers the user's question, before any supporting detail. "
        "In Actionable Recommendations, order recommendations by expected "
        "business impact and state the expected impact of each one "
        "explicitly.\n\n"
        "Structure the report with these Markdown headings, in this order:\n"
        "## Executive Summary\n"
        "## Business Objective\n"
        "## Dataset Overview\n"
        "## Data Quality Assessment\n"
        "## Exploratory Data Analysis\n"
        "## Descriptive Statistics\n"
        "## KPI Summary\n"
        "## Business Analysis\n"
        "## Trend Analysis\n"
        "## Root Cause Analysis\n"
        "## Statistical Findings\n"
        "## Forecasting Results\n"
        "## Segmentation Findings\n"
        "## Visualization Interpretation\n"
        "## Key Insights\n"
        "## Actionable Recommendations\n"
        "## Risks\n"
        "## Limitations\n"
        "## Conclusion\n\n"
        "In 'Visualization Interpretation', refer to the generated charts "
        "by their number (e.g. 'Chart 1 shows...') using the chart list "
        "below. Keep each section focused — a few sentences or bullet "
        "points backed by real numbers, not filler text. The report must "
        "directly answer the user's original question.\n\n"
        f"User question: {state['user_query']}\n\n"
        f"Dataset profile:\n{json.dumps(state['dataset_profile'], indent=2, default=str)}\n\n"
        f"Analysis plan:\n{state['analysis_plan']}\n\n"
        f"Analysis results:\n{state['analysis_results']}\n\n"
        f"Insights:\n{state['insights']}\n\n"
        f"Generated charts:\n{chart_list}"
    )
    report = llm_report.invoke(prompt).content
    return {
        "final_report": report,
        "messages": [AIMessage(content=report)],
    }


# GRAPH DEFINITION(NOW WE BUILD THE FINAL GRAPH)
graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner", planner_agent)
graph_builder.add_node("dataset_understanding", dataset_understanding_agent)
graph_builder.add_node("intent_router", intent_router)
graph_builder.add_node("follow_up_response", follow_up_response_agent)
graph_builder.add_node("analysis_planning", analysis_planning_agent)
graph_builder.add_node("data_analysis", data_analysis_agent)
graph_builder.add_node("visualization", visualization_agent)
graph_builder.add_node("insight_generation", insight_generation_agent)
graph_builder.add_node("report_generation", report_generation_agent)

graph_builder.add_edge(START, "planner")
graph_builder.add_conditional_edges("planner", route_after_planner, { "dataset_understanding": "dataset_understanding", "intent_router": "intent_router",},)
graph_builder.add_edge("dataset_understanding", "intent_router")
graph_builder.add_conditional_edges("intent_router", route_after_intent, { "follow_up_response": "follow_up_response", "analysis_planning": "analysis_planning",},)
graph_builder.add_edge("follow_up_response", END)
graph_builder.add_edge("analysis_planning", "data_analysis")
graph_builder.add_edge("data_analysis", "visualization")
graph_builder.add_edge("visualization", "insight_generation")
graph_builder.add_edge("insight_generation", "report_generation")
graph_builder.add_edge("report_generation", END)

checkpointer = InMemorySaver()
compiled_graph = graph_builder.compile(checkpointer=checkpointer)#GRAPH COMPILED