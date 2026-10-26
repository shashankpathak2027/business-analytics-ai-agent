# Analytica — Enterprise Business Analytics AI Agent

An AI data analyst that takes a single CSV upload and answers open-ended
business questions with evidence-backed analysis: descriptive statistics,
business and statistical analysis, forecasting and segmentation where
appropriate, root-cause investigation, up to four business-value-ranked
charts per analysis, and a consulting-style executive report — with working
conversation memory for follow-up questions.

Built with LangGraph, LangChain, Google Gemini, FastAPI, and LangSmith.

## Features

- Autonomous multi-step analysis: dataset profiling → analysis planning →
  code-driven analysis → visualization → insight generation → executive report
- Follow-up questions answered from the existing analysis instead of
  re-running the full pipeline
- At most 4 charts per analysis, each with a title and business interpretation
- Graceful degradation: a failed analysis step is retried once, then noted
  honestly in plain language — raw Python errors are never shown to the user
- Full LangSmith tracing: every node, LLM call, and tool call is a named,
  timed span
- Conversation memory via LangGraph's `InMemorySaver`, keyed by thread ID —
  no external database
- A minimal HTML/CSS/JS demo frontend, decoupled from the API 

## Architecture

```
START -> planner
planner --[profile empty]--> dataset_understanding --> intent_router
planner --[profile exists]-------------------------------> intent_router
intent_router --[follow_up]--> follow_up_response --> END
intent_router --[new_analysis]--> analysis_planning -> data_analysis
    -> visualization -> insight_generation -> report_generation -> END
```

| Agent | Responsibility |
|---|---|
| Planner | Reads the latest message, decides whether the dataset needs profiling |
| Dataset Understanding | Profiles the CSV and summarizes it for the user |
| Intent Router | Classifies each message as `new_analysis` or `follow_up` |
| Follow-Up Response | Answers follow-ups from existing analysis, no new code execution |
| Analysis Planning | Produces a concrete, dataset-specific execution plan |
| Data Analysis | Generates and executes pandas/numpy/scipy code, with one repair retry |
| Visualization | Chooses up to 4 high-value charts, each with a title + explanation |
| Insight Generation | Derives evidence-backed business insights |
| Report Generation | Compiles the full executive report |

## Project Structure

```
business-analytics-agent/
├── backend/
│   ├── config.py       # Env vars, LangSmith setup, specialized LLM clients
│   └── graph.py         # State, tools, all agent nodes, LangGraph workflow
├── api/
│   └── main.py            # FastAPI: upload, query, streaming, thread handling
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js               # Demo UI, talks to the API over fetch()
├── data/uploads/              # Uploaded CSVs (created at runtime)
├── outputs/charts/             # Generated chart PNGs (created at runtime)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone the project and create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Then fill in `.env`:
   - `GOOGLE_API_KEY` — a Google Gemini API key (get a free one at
     [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
   - `LANGCHAIN_API_KEY` — a LangSmith API key (optional but recommended —
     get a free one at [smith.langchain.com](https://smith.langchain.com))

## Running the Backend

From the project root (imports are rooted here — `backend.*` and `api.*`):

```bash
uvicorn api.main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

## Running the Frontend

The frontend is static HTML/CSS/JS — serve the `frontend/` folder with any
static file server, for example:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500`. If the API isn't running at
`http://localhost:8000`, update `API_BASE_URL` at the top of `app.js`.

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a CSV, returns a `thread_id` |
| `POST` | `/api/query` | Ask a question for a given `thread_id` (blocking) |
| `GET` | `/api/health` | Health check |

`POST /api/query` request body:
```json
{ "thread_id": "...", "question": "Why are sales decreasing?" }
```

Response:
```json
{
  "thread_id": "...",
  "analysis_plan": "...",
  "analysis_results": "...",
  "insights": "...",
  "final_report": "...",
  "charts": [
    { "url": "/charts/chart_xxxx.png", "title": "...", "explanation": "..." }
  ]
}
```

## Memory & Follow-Up Questions

Each uploaded CSV starts a new conversation thread (`thread_id`), backed by
LangGraph's `InMemorySaver`. Every question for that thread reuses the same
state — the dataset profile, analysis plan, results, insights, charts, and
report all persist across turns for as long as the server process is
running. The Intent Router decides, per message, whether a question needs
fresh analysis or can be answered from what's already there, so asking
"summarize the report" after "analyze my dataset" doesn't re-run the
pipeline from scratch.

Memory is in-process only — restarting the server clears all threads. This
is intentional (no external database, per the project's requirements); for a
persistent deployment, LangGraph's checkpointer could be swapped for a
different backend without changing any node logic.

## Observability

Every agent node is wrapped in `@traceable` and every graph run is tagged
with the thread ID and the user's question, so a LangSmith trace can be
filtered by thread and inspected node-by-node — inputs, outputs, and timing
for every LLM call and tool call. Set `LANGCHAIN_API_KEY` in `.env` to
enable it; the app runs fine without it, just without tracing.

## Notes & Limitations

- The frontend website is made using AI and it is just a demonstration that 
  the API i created can be connected to any Frontend(app or website) with 
  minimal changes. Therefore the main components of this project are the 
  Backend and the API.
- Forecasting uses simple trend extrapolation (`numpy.polyfit` / rolling
  means) computed by the Data Analysis Agent — no dedicated forecasting
  library is installed.
- Chart generation and analysis code execution both run in a local Python
  REPL (`langchain_experimental.utilities.PythonREPL`) with no sandboxing
  beyond the process itself. 
- Uploaded CSVs and generated charts are stored on local disk
  (`data/uploads/`, `outputs/charts/`) and are never cleaned up
  automatically.