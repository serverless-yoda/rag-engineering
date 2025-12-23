# agents/planner.py

"""
🧠 PlannerAgent: Generates a structured multi-agent execution plan.

This agent analyzes a high-level goal and produces a step-by-step plan
using available agents (Librarian, Researcher, Writer). It uses the LLM
to synthesize the plan in JSON format.
"""

import json
from ..abstractions.llm_provider import LLMProvider


class PlannerAgent:
    """
    PlannerAgent: Uses the LLM to generate a structured execution plan.

    The plan is a list of steps, each specifying:
    - step number
    - agent name
    - input parameters

    Example output:
    [
        {"step": 1, "agent": "Librarian", "input": {"intent": "suspenseful narrative blueprint"}},
        {"step": 2, "agent": "Researcher", "input": {"topic": "Apollo 11"}},
        {"step": 3, "agent": "Writer", "input": {"blueprint": "STEP_1_OUTPUT", "facts": "STEP_2_OUTPUT"}}
    ]
    """

    def __init__(self, pipeline):
        """
        Initialize with access to the pipeline's LLM provider.
        """
        self.llm: LLMProvider = pipeline.llm

    async def create_plan(self, goal: str, capabilities: str) -> list:
        """
        Generate a multi-step plan using the LLM.

        Args:
            goal: The user's high-level goal (e.g., "Write a suspenseful story")
            capabilities: A string describing available agents and their inputs

        Returns:
            A list of steps (dicts) forming the execution plan

        Raises:
            ValueError if the LLM response is not a valid JSON list
        """
        system_prompt = f"""
        You are the strategic planner of a multi-agent AI system.
        Your job is to break down the user's goal into a structured execution plan.

        --- AVAILABLE AGENTS ---
        {capabilities}
        --- END AGENTS ---

        INSTRUCTIONS:
        - Return a JSON list of steps.
        - Each step must include: step number, agent name, input dictionary.
        - Use context chaining: reference previous outputs using STEP_X_OUTPUT.
        - Do not include explanations or extra text.

        Example goal: "Write a suspenseful story about Apollo 11"
        Example plan:
        [
            {{"step": 1, "agent": "Librarian", "input": {{"intent": "suspenseful narrative blueprint"}}}},
            {{"step": 2, "agent": "Researcher", "input": {{"topic": "Apollo 11"}}}},
            {{"step": 3, "agent": "Writer", "input": {{"blueprint": "STEP_1_OUTPUT", "facts": "STEP_2_OUTPUT"}}}}
        ]
        """

        try:
            response = await self.llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": goal}
                ],
                temperature=0.3
            )
            plan = json.loads(response)

            # Handle wrapped responses
            if isinstance(plan, dict):
                if "plan" in plan and isinstance(plan["plan"], list):
                    return plan["plan"]
                elif "steps" in plan and isinstance(plan["steps"], list):
                    return plan["steps"]
                else:
                    raise ValueError("Planner returned a dict, but missing 'plan' or 'steps' key.")

            if not isinstance(plan, list):
                raise ValueError("Planner did not return a valid JSON list.")

            return plan

        except Exception as e:
            raise ValueError(f"Planner failed to generate a valid plan: {e}")
