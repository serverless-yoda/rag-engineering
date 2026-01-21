# rag/models/__init__.py

"""
Models package for RAG system data structures.

Exports all configuration objects and result types used throughout the B.I.S.A. pipeline.
"""
from .env import env_settings
from .config import RAGConfig
from .agent_response import AgentResponse
from .exceptions import PipelineError, SearchError, GenerationError, IngestionError, SafetyCheckError, PlanningError, AgentExecutionError
from .types import (
    ChunkingConfig,
    IngestionResult,
    SearchResult,
    JsonDict,
)
from .execution_trace import ExecutionTrace
from .log_config import (
    request_id_ctx, 
    agent_name_ctx, 
    pipeline_stage_ctx,
    setup_json_logging
)

__all__ = [
    "ChunkingConfig",
    "IngestionResult",  
    "SearchResult",
    "JsonDict",
    "RAGConfig",
    "env_settings",
    "AgentResponse",
    "PipelineError",
    "SearchError",  
    "GenerationError",
    "IngestionError",
    "SafetyCheckError",
    "PlanningError",
    "AgentExecutionError",
    "ExecutionTrace"
    "request_id_ctx", 
    "agent_name_ctx", 
    "pipeline_stage_ctx"
    "setup_json_logging"
]