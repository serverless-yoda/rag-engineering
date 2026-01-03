# engine/context_engine.py

"""Context Engine: Multi-agent orchestration with planning."""
import copy
import logging
from ..agents.registry import AgentRegistry
from ..agents.planner import PlannerAgent
from ..models import AgentResponse, PipelineError


# Ensure agent modules are loaded
from ..agents.researcher import ResearcherAgent
from ..agents.writer import WriterAgent
from ..agents.summarizer import SummarizerAgent
from ..agents.librarian import LibrarianAgent

from typing import Dict
class ContextEngine:
    def __init__(self, searcher,generator , content_safety=None):
        
        self.registry = AgentRegistry(
            searcher=searcher,
            generator=generator,
            content_safety=content_safety
        )

        self.planner = PlannerAgent(generator)
    
    async def execute(self, goal: str):
        try:

            # Phase 1: Plan
            plan = await self.planner.create_plan(
                goal, 
                self.registry.get_capabilities(),
                known_agents=list(self.registry._registry.keys())
            )
            logging.info(f"Executed Planner: {plan}")
            
            # Phase 2: Execute with context chaining
            state = {}
            for step in plan:
                agent = self.registry.get(step['agent'])
                resolved_input = self._resolve_dependencies(step['input'], state)
                mcp_input = {"content": resolved_input}
                mcp_output: AgentResponse= await agent.execute(mcp_input)
                state[f"STEP_{step['step']}_OUTPUT"] = mcp_output.content
                print(f"Executed Step {step['step']} with agent {step['agent']}")
                print(f"Input: {mcp_input}")
                print(f"Output: {mcp_output}")

                
                # Check if step was blocked            
                if mcp_output.status == "blocked":
                    logging.warning(f"⚠️ Workflow blocked at step {step['step']}")
                    return mcp_output.content
            
            return state[f"STEP_{len(plan)}_OUTPUT"]
        except PipelineError as e:              
            logging.error(f"❌ Pipeline error: {e}")
            return {"error": str(e)}
  
    
    def _resolve_dependencies(self, input_params: Dict, state: Dict) -> Dict:
        """
        Resolve STEP_X_OUTPUT references with validation.
        Handles nested content extraction (e.g., {'facts': 'text'} -> 'text').
        """
        resolved = copy.deepcopy(input_params)
        
        for key, value in resolved.items():
            if isinstance(value, str) and value.startswith("STEP_"):
                # Validate dependency exists
                if value not in state:
                    available = [k for k in state.keys() if k.startswith("STEP_")]
                    raise ValueError(
                        f"Unresolved dependency: '{value}' in input parameter '{key}'\n"
                        f"Available outputs: {available}"
                    )
                
                resolved_value = state[value]
                
                # Smart unwrapping: if resolved value is a dict with a single content key, extract it
                if isinstance(resolved_value, dict):
                    # Try common content keys
                    for content_key in ['output', 'facts', 'summary', 'blueprint']:
                        if content_key in resolved_value:
                            resolved[key] = resolved_value[content_key]
                            break
                    else:
                        # No common key, pass whole dict
                        resolved[key] = resolved_value
                else:
                    resolved[key] = resolved_value
        
        return resolved
