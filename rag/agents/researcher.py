# agents/researcher.py

"""
🔍 ResearcherAgent: Synthesizes factual information from the knowledge store.

This agent performs semantic search over the 'KnowledgeStore' namespace
and uses the LLM to summarize the retrieved chunks into concise facts.
"""

import logging
from ..utils import sanitize_input
from .base_agent import BaseAgent
from ..interfaces import SearchProvider, GenerationProvider
from ..models import AgentResponse, AgentExecutionError
from ..agents.registry import AgentRegistry

@AgentRegistry.register(
    name="researcher",
    capabilities="Retrieves and synthesizes factual information on a topic.",
    required_inputs=["topic"]
)
class ResearcherAgent(BaseAgent):
    def __init__(self, searcher: SearchProvider, generator: GenerationProvider):
        """
        Initialize with access to the RAGPipeline.
        """
        self.searcher = searcher
        self.generator = generator


    async def setup(self):
        logging.info(f"[{self.__class__.__name__}] Setup started.")
        # Add any initialization logic here
        logging.info(f"[{self.__class__.__name__}] Setup completed.")

    async def teardown(self):
        logging.info(f"[{self.__class__.__name__}] Teardown started.")
        # Add any cleanup logic here
        logging.info(f"[{self.__class__.__name__}] Teardown completed.")

    async def execute(self, mcp_message):
        """
        Execute the researcher agent.

        Args:
            mcp_message: Dict with 'content' containing a 'topic' key.

        Returns:
            Dict with synthesized 'facts' as content.
        """
        self.validate_input(mcp_message['content'], ['topic'])

        
        try:
            topic = mcp_message['content']['topic']
            results = await self.searcher.search(query=topic, namespace="KnowledgeStore", top_k=3)

            chunks = [str(r.chunk) for r in results if r.chunk]
            if not chunks:
                return AgentResponse(
                    sender="Researcher",
                    content={},
                    status="error",
                    error_message="No valid chunks found"
                )

            context = "\n\n".join(chunks)
            system_prompt = (
                "You are an expert research synthesis AI.\n"
                "Synthesize the provided source texts into a concise, bullet-pointed summary "
                "relevant to the user's topic. Focus strictly on the facts provided in the sources. "
                "Do not add outside information."
            )
            facts = await self.generator.generate(question=topic, context=context, system_prompt=system_prompt)

            return AgentResponse(
                sender="Researcher",
                content={"facts": facts}
            )
        except Exception as e:
            # return AgentResponse(
            #     sender="Researcher",
            #     content={},
            #     status="error",
            #     error_message=str(e)
            # )
            raise AgentExecutionError(f"ResearcherAgent execution failed:{e}")

