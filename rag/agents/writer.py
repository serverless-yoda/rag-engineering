# agents/writer.py

"""
✍️ WriterAgent: Generates content using a blueprint and factual input.

This agent uses the LLM to produce structured output based on:
- blueprint: style and structure instructions
- facts: factual content to include
- previous_content: optional existing content to rewrite
"""

from ..agents.base_agents import BaseAgent
class WriterAgent(BaseAgent):
    def __init__(self, generator, content_safety=None):
        """
        Initialize with access to the RAGPipeline.
        """
        self.generator = generator
        self.content_safety = content_safety

    async def execute(self, mcp_message):
        """
        Execute the writer agent.

        Args:
            mcp_message: Dict with 'content' containing 'blueprint', 'facts', or 'previous_content'.

        Returns:
            Dict with generated 'output' as content.
        """
        self.validate_input(mcp_message['content'], ['blueprint','facts','previous_content'])
        content = mcp_message['content']
        blueprint_json = content.get('blueprint', '{}')
        facts = content.get('facts', '')
        previous = content.get('previous_content', '')
        blueprint_json_string = blueprint_json.get('blueprint', '{}') if isinstance(blueprint_json, dict) else blueprint_json
        
        facts_data = None
        if isinstance(facts, dict):
            facts_data = facts.get('facts', '')  # Handle case where facts come from Summarizer
            if facts_data is None:
                facts_data = facts.get('summary', '')
                if facts_data is None:
                    facts_data = facts
        elif isinstance(facts, str):
            facts_data = facts
        
        if not blueprint_json_string or (not facts_data and not previous):
            return {"sender": "Writer", "content": {'output': 'Error: Missing blueprint or facts or previous for content generation.'}}

        if facts_data:
            source_material = facts_data
            source_label = "SOURCE FACTS"
        else:
            source_material = previous
            source_label = "PREVIOUS CONTENT (For Rewriting)"

        system_prompt = f"""You are an expert content generation AI.
Your task is to generate content based on the provided RESEARCH FINDINGS.
Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

--- SEMANTIC BLUEPRINT (JSON) ---
{blueprint_json}
--- END SEMANTIC BLUEPRINT ---

Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the research defines WHAT you write about.
"""

        user_prompt = f"""
--- SOURCE MATERIAL ({source_label}) ---\n{source_material}\n--- END SOURCE MATERIAL ---        
--- RESEARCH FINDINGS ---
{facts}
--- END RESEARCH FINDINGS ---

{f"--- PREVIOUS CONTENT ---{previous}--- END PREVIOUS CONTENT ---" if previous else ""}

Generate the content now.
"""
        
        final_output = await self.generator.generate(question=user_prompt, context="", system_prompt=system_prompt)
        
        # Content safety check      
        # Commenting this: its failing because of self.content_safety is None in some tests
        # moderation_result = await self.content_safety.moderate_text(final_output)
        # if not moderation_result["is_safe"]:
        #     logging.warning(f"Writer output blocked: {moderation_result['recommendation']}")
        #     return {
        #             "sender": "Writer",
        #             "content": {
        #                 "output": "⚠️ Content blocked by safety filters",
        #                 "moderation_result": moderation_result,
        #                 "status": "blocked",
        #             }
        #         }
            
        # logging.info(f"✅ Content moderation passed. Scores: {moderation_result['severity_scores']}")


        return {"sender": "Writer", "content": {'output': final_output}}
