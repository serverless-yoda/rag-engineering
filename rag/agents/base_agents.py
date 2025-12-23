
# agents/base_agents.py

"""Base abstraction for all agents."""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Abstract base class for agents in the multi-agent system."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    @abstractmethod
    async def execute(self, mcp_message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic and return MCP message."""
        pass
