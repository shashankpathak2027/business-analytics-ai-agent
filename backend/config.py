
"""
config.py:
Initialization module for the Enterprise Business Analytics AI Agent.
Every other backend/API file imports what it needs from this module.
"""

import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
load_dotenv()

# API KEYS 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY 

# LANGSMITH OBSERVABILITY
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true") 
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash")

llm_router = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0,
    max_output_tokens=512,
    thinking_budget=0, 
    google_api_key=GOOGLE_API_KEY,)

llm_reasoning = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0.2,
    max_output_tokens=3072,
    thinking_budget=512,
    google_api_key=GOOGLE_API_KEY,)

llm_coder = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0,
    max_output_tokens=4096,
    thinking_budget=100,
    google_api_key=GOOGLE_API_KEY,)

llm_report = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0.3,
    max_output_tokens=8192,
    thinking_budget=512,
    google_api_key=GOOGLE_API_KEY,)

# PROJECT DIRECTORIES PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

