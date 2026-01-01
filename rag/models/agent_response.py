# models/agent_response.py
"""Data model for agent responses in the RAG system."""

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

@dataclass
class AgentResponse:
    sender: str
    content: Dict[str, Any]
    status: Literal["success", "error", "blocked"] = "success"
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        return self.status == "success"
