# engine/context_engine.py

"""Context Engine: Multi-agent orchestration with planning."""
import copy
import logging
from ..agents.registry import AgentRegistry
from ..agents.planner import PlannerAgent

class ContextEngine:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.registry = AgentRegistry(pipeline)
        self.planner = PlannerAgent(pipeline)
    
    async def execute(self, goal: str):
        # Phase 1: Plan
        plan = await self.planner.create_plan(
            goal, 
            self.registry.get_capabilities()
        )
        logging.info(f"Executed Planner: {plan}")
        
        # Phase 2: Execute with context chaining
        state = {}
        for step in plan:
            agent = self.registry.get(step['agent'])
            resolved_input = self._resolve_dependencies(step['input'], state)
            mcp_input = {"content": resolved_input}
            mcp_output = await agent.execute(mcp_input)
            state[f"STEP_{step['step']}_OUTPUT"] = mcp_output['content']
            logging.info(f"Executed Step {step['step']} with agent {step['agent']}")
            logging.info(f"Input: {mcp_input}")
            logging.info(f"Output: {mcp_output}")
        
        return state[f"STEP_{len(plan)}_OUTPUT"]
    
    def _resolve_dependencies(self, input_params, state):
        """Replace STEP_X_OUTPUT references with actual data."""
        resolved = copy.deepcopy(input_params)
        for key, value in resolved.items():
            if isinstance(value, str) and value.startswith("STEP_"):
                resolved[key] = state.get(value, value)
        return resolved
