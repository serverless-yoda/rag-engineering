
# agents/summarizer.py

"""
📝 SummarizerAgent: Condenses large bodies of text into concise summaries tailored to a specific objective.

This agent uses the LLM to:
- Interpret a user-defined summarization goal
- Extract and distill key information from the input text
- Return a focused, goal-aligned summary
"""

class SummarizerAgent:
    def __init__(self, pipeline):
        """
        Initialize the SummarizerAgent with access to the RAGPipeline.

        The pipeline provides access to the LLM and other shared services.
        """
        self.pipeline = pipeline

    async def execute(self, mcp_message):
        """
        Execute the summarization task.

        This method extracts the text and objective from the MCP message,
        constructs a prompt for the LLM, and returns the generated summary.

        Args:
            mcp_message (dict): A message containing:
                - 'text_to_summarize': The full input text
                - 'summary_objective': A description of what the summary should focus on

        Returns:
            dict: An MCP-style response containing the summary or an error message.
        """
        try:
            #print("SummarizerAgent received message:", mcp_message)
            # Extract required fields from the input message
            text_to_summarize = mcp_message['content'].get('text_to_summarize', "")
            summary_objective = mcp_message['content'].get('summary_objective', "")
            #print(f"Text to summarize : {text_to_summarize}...")
            #print(f"Summary objective: {summary_objective}")

            # Validate input presence
            if not text_to_summarize or not summary_objective:
                raise ValueError("Both 'text_to_summarize' and 'summary_objective' must be provided..")

            # Define the system prompt to guide the LLM's behavior
            system_prompt = (
                "You are an expert summarization AI. Your task is to reduce the provided text to its essential points, "
                "guided by the user's specific objective. The summary must be concise, accurate, and directly address the stated goal."
            )

            # Construct the user prompt with both the objective and the text
            user_prompt = (
                f"--- OBJECTIVE ---\n{summary_objective}\n\n"
                f"--- TEXT TO SUMMARIZE ---\n{text_to_summarize}\n--- END TEXT ---\n\n"
                "Generate the summary now."
            )

            # Call the LLM via the pipeline's generate method
            final_output = await self.pipeline.generate(
                question=user_prompt,
                context="",
                system_prompt=system_prompt
            )

            # Return the summary in MCP format
            return {"sender": "Summarizer", "content": {'output': final_output}}

        except Exception as e:
            # Return a structured error message in case of failure
            return {"sender": "Summarizer", "content": {'output': f'Error during summarization: {str(e)}'}}
