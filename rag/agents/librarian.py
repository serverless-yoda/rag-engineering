# rag/agents/librarian.py

"""Librarian agent: Retrieves semantic blueprints."""
import json
import logging
from .base_agent import BaseAgent
from ..models import AgentResponse, AgentExecutionError
from ..agents.registry import AgentRegistry

@AgentRegistry.register(
    name="librarian",
    capabilities="Retrieves Semantic Blueprints (style/structure instructions).",
    required_inputs=["intent"]
)

class LibrarianAgent(BaseAgent):
    def __init__(self, searcher):
        self.searcher = searcher


    async def setup(self):
        logging.info(f"[{self.__class__.__name__}] Setup started.")
        # Add any initialization logic here
        logging.info(f"[{self.__class__.__name__}] Setup completed.")

    async def teardown(self):
        logging.info(f"[{self.__class__.__name__}] Teardown started.")
        # Add any cleanup logic here
        logging.info(f"[{self.__class__.__name__}] Teardown completed.")
 
    async def execute(self, mcp_message):
        self.validate_input(mcp_message['content'], ['intent'])

        try:

            intent = mcp_message['content']['intent']
            results = await self.searcher.search(
                query=intent,
                namespace="ContextLibrary",
                top_k=1
            )
            
            if results:
                blueprint_json = results[0].metadata.get('blueprint_json', '{}')
                content = {'blueprint': blueprint_json}
            else:
                content = {'blueprint': json.dumps({'instruction': 'Generate neutral content'})}
            
            return AgentResponse(
                    sender="Librarian",
                    content={"content": content}
                )
        except Exception as e:
            # return AgentResponse(
            #                 sender="Librarian",
            #                 content={},
            #                 status="error",
            #                 error_message=str(e)
            #             )
            raise AgentExecutionError(f"LibrarianAgent execution failed: {e}")