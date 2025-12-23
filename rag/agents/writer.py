# agents/writer.py

"""
✍️ WriterAgent: Generates content using a blueprint and factual input.

This agent uses the LLM to produce structured output based on:
- blueprint: style and structure instructions
- facts: factual content to include
- previous_content: optional existing content to rewrite
"""

class WriterAgent:
    def __init__(self, pipeline):
        """
        Initialize with access to the RAGPipeline.
        """
        self.pipeline = pipeline

    async def execute(self, mcp_message):
        """
        Execute the writer agent.

        Args:
            mcp_message: Dict with 'content' containing 'blueprint', 'facts', or 'previous_content'.

        Returns:
            Dict with generated 'output' as content.
        """
        #print("WriterAgent received message:", mcp_message)
        content = mcp_message['content']
        blueprint_json = content.get('blueprint', '{}')
        #print(f"Blueprint JSON: {blueprint_json}")
        facts = content.get('facts', '')
        #print(f"\nFacts: {facts}")
        previous = content.get('previous_content', '')
        #print(f"\nPrevious Content: {previous}")
        blueprint_json_string = blueprint_json.get('blueprint', '{}') if isinstance(blueprint_json, dict) else blueprint_json
        #print(f"\nUsing Blueprint: {blueprint_json_string}")

        facts_data = None
        if isinstance(facts, dict):
            facts_data = facts.get('facts', '')  # Handle case where facts come from Summarizer
            if facts_data is None:
                facts_data = facts.get('summary', '')
                if facts_data is None:
                    facts_data = facts
        elif isinstance(facts, str):
            facts_data = facts
        
        print(f"\nFacts data: {facts_data}")

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
        
        final_output = await self.pipeline.generate(question=user_prompt, context="", system_prompt=system_prompt)
        return {"sender": "Writer", "content": {'output': final_output}}
