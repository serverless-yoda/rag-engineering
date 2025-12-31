
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

    def validate_input(self, content: Dict[str, Any], required_keys: list) -> bool:
        """Validate that required keys are present in the content."""
        for field in required_keys:
            if field not in content:
                raise ValueError(f"Missing required field: {field}")
            
        