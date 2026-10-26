# FastAPI app: CSV upload, query endpoint, thread/session handling, streaming

"""
main.py
-------
FastAPI layer for the Enterprise Business Analytics AI Agent.
This file only depends on backend/graph.py and backend/config.py and can be
connected to any frontend without changes.
"""

import os
import uuid
import logging
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from backend.config import UPLOAD_DIR, CHARTS_DIR
from backend.graph import compiled_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytics_agent")


app = FastAPI(title="Enterprise Business Analytics AI Agent")

app.add_middleware( #SO THAT WE CAN ACCEPT REQUESTS FROM ANY ORIGINS, BECAUSE OUR BACKEND AND FRONTEND WILL BE HOSTED AT DIFFERENT LOCATIONS
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")

SESSION_CSV_PATHS: Dict[str, str] = {}


class QueryRequest(BaseModel): #VALIDATING THE INPUT AND CREATING A PYTHON OBJECT
    thread_id: str
    question: str


def _charts_payload(chart_paths) -> List[Dict[str, str]]:
    """Converts internal chart records (local file path + title +
    explanation, as produced by the Visualization Agent) into the URL-based
    payload the frontend renders."""
    charts = []
    for chart in chart_paths or []:
        charts.append(
            {
                "url": f"/charts/{os.path.basename(chart['path'])}",
                "title": chart.get("title", ""),
                "explanation": chart.get("explanation", ""),
            }
        )
    return charts


def _graph_config(thread_id: str, question: str) -> dict:
    """Builds the LangGraph run config, including LangSmith run naming, tags,
    and metadata so every query is easy to find and filter in a trace."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "Analytics Query",
        "tags": [f"thread:{thread_id}"],
        "metadata": {"thread_id": thread_id, "question": question},
    }

#WE ONLY HAVE 2 ENDPOINTS, ONE TO UPLOAD CSV AND ANOTHER TO ASK QUESTIONS.
@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Uploads a CSV file and starts a new conversation thread for it."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    thread_id = str(uuid.uuid4())
    saved_filename = f"{thread_id}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        f.write(await file.read())

    SESSION_CSV_PATHS[thread_id] = saved_path

    return {
        "thread_id": thread_id,
        "filename": file.filename,
        "message": "File uploaded successfully. You can now ask questions about this dataset.",
    }


@app.post("/api/query")
async def query_agent(request: QueryRequest):
    """Runs a user question through the full LangGraph workflow and returns
    the final results (blocking, non-streaming)."""
    csv_path = SESSION_CSV_PATHS.get(request.thread_id)
    if not csv_path:
        raise HTTPException(
            status_code=404, detail="Unknown thread_id. Please upload a CSV first."
        )

    graph_input = {
        "messages": [HumanMessage(content=request.question)],
        "csv_path": csv_path,
    }
    config = _graph_config(request.thread_id, request.question)

    try:
        result = compiled_graph.invoke(graph_input, config)
    except Exception:
        logger.exception("Graph execution failed for thread_id=%s", request.thread_id)
        raise HTTPException(
            status_code=500,
            detail="The agent could not process your question. Please try again.",
        )

    return {
        "thread_id": request.thread_id,
        "analysis_plan": result.get("analysis_plan", ""),
        "analysis_results": result.get("analysis_results", ""),
        "insights": result.get("insights", ""),
        "final_report": result.get("final_report", ""),
        "charts": _charts_payload(result.get("chart_paths")),
    }


@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}