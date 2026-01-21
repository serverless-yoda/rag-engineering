# rag/models/execution_trace.py

"""Data model for execution trace of agents in the RAG pipeline."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ExecutionTrace:
    step: int
    agent: str
    input_summary: str
    output_summary: str
    status: str
    error_message: Optional[str]
    duration_ms: float
    timestamp: datetime
