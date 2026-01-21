# rag/agents/base_agent.py

"""Base abstraction for all agents."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models import AgentResponse, AgentExecutionError
class BaseAgent(ABC):
    """Abstract base class for agents in the multi-agent system."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    @abstractmethod
    async def execute(self, mcp_message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic and return MCP message."""
        pass

    def validate_input(self, content: Dict[str, Any], required_keys: list) -> bool:
        """Validate that required keys are present in the content."""
        for field in required_keys:
            if field not in content:
                raise ValueError(f"Missing required field: {field}")
            
    
    async def setup(self):
        """Optional setup hook called before execution begins."""
        pass

    async def teardown(self):
        """Called on shutdown or context exit"""
        pass

    
    async def on_error(self, error: Exception, context: Dict[str, Any]) -> Optional[AgentResponse]:
        """
        Error recovery hook. Return AgentResponse to continue, None to propagate error.
        """
        raise AgentExecutionError(f"{self.__class__.__name__} failed: {error}")

            
        