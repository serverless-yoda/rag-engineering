# agents/librarian.py

"""Librarian agent: Retrieves semantic blueprints."""
import json
from ..agents.base_agents import BaseAgent

class LibrarianAgent(BaseAgent):
    def __init__(self, searcher):
        self.searcher = searcher
    
    async def execute(self, mcp_message):
        self.validate_input(mcp_message['content'], ['intent'])

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
        
        return {"sender": "Librarian", "content": content}
