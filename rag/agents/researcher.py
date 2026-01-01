# agents/researcher.py

"""
🔍 ResearcherAgent: Synthesizes factual information from the knowledge store.

This agent performs semantic search over the 'KnowledgeStore' namespace
and uses the LLM to summarize the retrieved chunks into concise facts.
"""

import json
from ..utils import sanitize_input
from ..agents.base_agents import BaseAgent
from ..interfaces import SearchProvider, GenerationProvider
from ..models import AgentResponse

class ResearcherAgent(BaseAgent):
    def __init__(self, searcher: SearchProvider, generator: GenerationProvider):
        """
        Initialize with access to the RAGPipeline.
        """
        self.searcher = searcher
        self.generator = generator


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
            system_prompt = "You are an expert research synthesis AI..."
            facts = await self.generator.generate(question=topic, context=context, system_prompt=system_prompt)

            return AgentResponse(
                sender="Researcher",
                content={"facts": facts}
            )
        except Exception as e:
            return AgentResponse(
                sender="Researcher",
                content={},
                status="error",
                error_message=str(e)
            )
