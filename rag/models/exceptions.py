
# models/exceptions.py
"""Custom exceptions for the RAG pipeline."""

class PipelineError(Exception):
    """Base exception for all RAG pipeline errors."""
    pass

class SearchError(PipelineError):
    """Raised when semantic search fails."""
    pass

class GenerationError(PipelineError):
    """Raised when LLM generation fails."""
    pass

class IngestionError(PipelineError):
    """Raised when document ingestion fails."""
    pass

class SafetyCheckError(PipelineError):
    """Raised when content safety moderation fails."""
    pass

class PlanningError(PipelineError):
    """Raised when PlannerAgent fails to produce a valid plan."""
    pass

class AgentExecutionError(PipelineError):
    """Raised when an agent fails during execution."""
    pass
