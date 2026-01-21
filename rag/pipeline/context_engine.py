# rag/pipeline/context_engine.py

"""Context Engine: Multi-agent orchestration with planning."""
import copy
import logging
import time
from ..agents.registry import AgentRegistry
from ..agents.planner import PlannerAgent
from ..models import AgentResponse, PipelineError, ExecutionTrace

# Ensure agent modules are loaded
from ..agents.researcher import ResearcherAgent
from ..agents.writer import WriterAgent
from ..agents.summarizer import SummarizerAgent
from ..agents.librarian import LibrarianAgent

from typing import Dict, Optional
from datetime import datetime


class ContextEngine:
    def __init__(self, searcher,generator , content_safety=None):
        
        self.registry = AgentRegistry(
            searcher=searcher,
            generator=generator,
            content_safety=content_safety
        )

        self.planner = PlannerAgent(generator)
        self.traces: list[ExecutionTrace] = []    
    
    async def execute(self, goal: str):
        try:
            # Phase 1: Plan
            plan = await self.planner.create_plan(
                goal,
                self.registry.get_capabilities(),
                known_agents=list(self.registry._registry.keys())
            )
            logging.info(f"Executed Planner: {plan}")

            # Phase 2: Setup agents
            used_agents = {step['agent'].lower() for step in plan}
            for agent_name in used_agents:
                agent = self.registry.get(agent_name)
                if hasattr(agent, "setup"):
                    await agent.setup()

            # Phase 3: Execute with context chaining
            state = {}
            for step in plan:
                start_time = time.time()
                agent_name = step['agent'].lower()
                agent = self.registry.get(agent_name)
                resolved_input = self._resolve_dependencies(step['input'], state)
                mcp_input = {"content": resolved_input}

                status = "success"
                error_message = None

                try:
                    mcp_output: AgentResponse = await agent.execute(mcp_input)
                except Exception as e:
                    status = "error"
                    error_message = str(e)
                    logging.error(f"❌ Agent '{agent_name}' failed at step {step['step']}: {e}")
                    if hasattr(agent, "on_error"):
                        recovery = await agent.on_error(e, {"step": step, "input": mcp_input})
                        if recovery:
                            mcp_output = recovery
                        else:
                            raise
                    else:
                        raise

                state[f"STEP_{step['step']}_OUTPUT"] = mcp_output.content

                # Add trace
                self.traces.append(ExecutionTrace(
                    step=step['step'],
                    agent=agent_name,
                    input_summary=str(mcp_input)[:200],
                    output_summary=str(mcp_output.content)[:200],
                    status=status,
                    error_message=error_message,
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now()
                ))

                print(f"Executed Step {step['step']} with agent {step['agent']}")
                if mcp_output.status == "blocked":
                    logging.warning(f"⚠️ Workflow blocked at step {step['step']}")
                    return mcp_output.content

            # Phase 4: Teardown agents
            for agent_name in used_agents:
                agent = self.registry.get(agent_name)
                if hasattr(agent, "teardown"):
                    await agent.teardown()

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


    def get_execution_report(self) -> str:
        lines = ["\n=== Execution Trace ==="]
        for trace in self.traces:
            lines.append(f"\nStep {trace.step}: {trace.agent} ({trace.duration_ms:.1f}ms)")
            lines.append(f"  Status: {trace.status}")
            if trace.error_message:
                lines.append(f"  Error: {trace.error_message}")
            lines.append(f"  Input: {trace.input_summary}")
            lines.append(f"  Output: {trace.output_summary}")
        return "\n".join(lines)
