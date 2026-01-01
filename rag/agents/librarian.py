# agents/librarian.py

"""Librarian agent: Retrieves semantic blueprints."""
import json
from ..agents.base_agents import BaseAgent
from ..models import AgentResponse

class LibrarianAgent(BaseAgent):
    def __init__(self, searcher):
        self.searcher = searcher
    
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
            
            #return {"sender": "Librarian", "content": content}
            return AgentResponse(
                    sender="Librarian",
                    content={"content": content}
                )
        except Exception as e:
            #return {"sender": "Librarian", "content": {'output': f'Error during retrieval: {str(e)}'}}
            return AgentResponse(
                            sender="Librarian",
                            content={},
                            status="error",
                            error_message=str(e)
                        )