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

        topic = mcp_message['content']['topic']
        results = await self.searcher.search(query=topic, namespace="KnowledgeStore", top_k=3)

        sanitized_chunks = []
        for r in results:
            chunk_text = str(r.chunk)
            try:
                sanitized_chunks.append(sanitize_input(chunk_text))
            except Exception:
                continue

        if not sanitized_chunks:
            return {
                "sender": "Researcher",
                "content": "Could not generate a reliable answer as retrieved data was suspect."
            }

        sources_text = "\n\n---\n\n".join(sanitized_chunks)
        system_prompt = (
            "You are an expert research synthesis AI.\n"
            "Synthesize the provided source texts into a concise, bullet-pointed summary "
            "relevant to the user's topic. Focus strictly on the facts provided in the sources. "
            "Do not add outside information."
        )

        result = await self.generator.generate(
            question=f"Topic: {topic}",
            context=sources_text,
            system_prompt=system_prompt
        )

        return {"sender": "Researcher", "content": {'facts': result}}

    async def _execute(self, mcp_message):
        """
        Execute the researcher agent.

        Args:
            mcp_message: Dict with 'content' containing a 'topic' key.

        Returns:
            Dict with synthesized 'facts' as content.
        """
        self.validate_input(mcp_message['content'], ['topic'])
        topic = mcp_message['content']['topic']
        results = await self.pipeline.search(
            query=topic,
            namespace="KnowledgeStore",
            top_k=3
        )

        sanitized_texts = []
        if results:
            source_chunks = []
            for r in results:
                chunk = r.chunk
                if isinstance(chunk, (dict, list)):
                    chunk_text = json.dumps(chunk, ensure_ascii=False)
                else:
                    chunk_text = str(chunk)

                sanitized_chunk = sanitize_input(chunk_text)
                if sanitized_chunk:
                    sanitized_texts.append(sanitized_chunk)
                    source_chunks.append(r.chunk)

            if not sanitized_texts:
                return {"sender": "Researcher", "content": "Could not generate a reliable answer as retrieved data was suspect."}

            #source_chunks = [r.chunk for r in results]
            sources_text = "\n\n---\n\n".join(
                json.dumps(chunk, ensure_ascii=False) if isinstance(chunk, (dict, list)) else str(chunk)
                for chunk in source_chunks
            )

            system_prompt = (
                "You are an expert research synthesis AI.\n"
                "Synthesize the provided source texts into a concise, bullet-pointed summary "
                "relevant to the user's topic. Focus strictly on the facts provided in the sources. "
                "Do not add outside information."
            )
            user_prompt = f"Topic: {topic}\n\nSources:\n{sources_text}"
            result = await self.pipeline.generate(question=user_prompt, context="", system_prompt=system_prompt)
            content = {'facts': result}
        else:
            content = {'facts': 'No knowledge data found'}

        return {"sender": "Researcher", "content": content}
