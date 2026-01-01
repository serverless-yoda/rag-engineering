# agents/planner.py

"""
🧠 PlannerAgent: Generates a structured multi-agent execution plan.

This agent analyzes a high-level goal and produces a step-by-step plan
using available agents (Librarian, Researcher, Writer). It uses the LLM
to synthesize the plan in JSON format.
"""

import json
import re
from typing import List, Dict
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

    def __init__(self, generator):
        self.generator = generator

    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        # Handle ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)

        # Strip leading/trailing whitespace
        return json.loads(response.strip())

    def _validate_plan_schema(self, plan: List[dict]) -> None:
        """Ensure each step has required fields."""
        if not isinstance(plan, list):
            raise ValueError(f"Plan must be a list, got {type(plan)}")

        for i, step in enumerate(plan):
            required = ['step', 'agent', 'input']
            missing = [f for f in required if f not in step]
            if missing:
                raise ValueError(f"Step {i} missing fields: {missing}. Step: {step}")

    async def create_plan(self, goal: str, capabilities: str) -> List[dict]:
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
            response = await self.generator.generate(
                question=goal,
                context="",
                system_prompt=system_prompt
            )

            plan_data = self._extract_json_from_response(response)

            # Handle wrapped dicts
            if isinstance(plan_data, dict):
                if "plan" in plan_data:
                    plan = plan_data["plan"]
                elif "steps" in plan_data:
                    plan = plan_data["steps"]
                else:
                    raise ValueError(f"Expected 'plan' or 'steps' key, got: {list(plan_data.keys())}")
            else:
                plan = plan_data

            self._validate_plan_schema(plan)
            return plan

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse plan as JSON: {e}\n\nRaw response:\n{response[:500]}")
        except Exception as e:
            raise ValueError(f"Planner failed to generate a valid plan: {e}")


