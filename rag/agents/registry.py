# agents/registry.py

"""
🧭 AgentRegistry: Central registry for agent lookup and capability description.

This module maps agent names to their handler classes and provides a structured
description of agent capabilities for use by the PlannerAgent.
"""

from ..agents.librarian import LibrarianAgent
from ..agents.researcher import ResearcherAgent
from ..agents.writer import WriterAgent
from ..agents.summarizer import SummarizerAgent

class AgentRegistry:
    """
    AgentRegistry: Maps agent names to their handlers and exposes capabilities.

    Used by:
    - ContextEngine to resolve agent handlers
    - PlannerAgent to understand available agents
    """
    def __init__(self, searcher, generator, content_safety=None):
        """
        Initialize the registry with access to the pipeline.

        Args:
            pipeline: RAGPipeline instance (used by agents)
        """
        self.registry = {
            "librarian": LibrarianAgent(searcher),
            "researcher": ResearcherAgent(searcher, generator),
            "writer": WriterAgent(generator, content_safety=content_safety),
            "summarizer": SummarizerAgent(generator),
        }

    def get(self, agent_name: str):
        """
        Retrieve the agent handler by name.

        Args:
            agent_name: Name of the agent (case-insensitive)

        Returns:
            Agent instance

        Raises:
            ValueError if agent is not found
        """
        agent_name = agent_name.lower()
        if agent_name not in self.registry:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")
        return self.registry[agent_name]

    def get_capabilities(self) -> str:
        """
        Return a structured description of agent capabilities.

        This is used by the PlannerAgent to generate valid plans.

        Returns:
            Multi-line string describing each agent's role and inputs
        """
        return """
Available Agents and their required inputs:

1. AGENT: Librarian
ROLE: Retrieves Semantic Blueprints (style/structure instructions).
INPUTS:
    - "intent": (String) A descriptive phrase of the desired style or format.
OUTPUT: The blueprint structure (JSON string).

2. AGENT: Researcher
ROLE: Retrieves and synthesizes factual information on a topic.
INPUTS:
    - "topic": (String) The subject matter to research.
OUTPUT: Synthesized facts (String).

3. AGENT: Summarizer
ROLE: Reduces large text to a concise summary based on a specific objective. Ideal for managing token counts before a generation step
INPUTS:
     - "text_to_summarize": (String/Reference) The long text to be summarized.
     - "summary_objective": (String) A clear goal for the summary (e.g., "Extract key technical specifications").
   OUTPUT: A dictionary containing the summary: {"summary": "..."}.

4. AGENT: Writer
ROLE: Generates or rewrites content by applying a Blueprint to source material.
INPUTS:
    - "blueprint": (String/Reference) The style instructions (usually from Librarian).
    - "facts": (String/Reference) Factual information (usually from Researcher). Use this for new content generation.
    - "previous_content": (String/Reference) Existing text (usually from a prior Writer step). Use this for rewriting/adapting content.
OUTPUT: The final generated text (String).


"""
