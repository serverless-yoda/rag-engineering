# /orchestrator.py

"""
📌 Example usage of the RAGPipeline orchestrator.

This script demonstrates:
1. Setting up the index and ingesting documents
2. Asking a question and retrieving an answer
3. Accessing search results directly
4. Executing a multi-agent contextual generation workflow

Run this script with: python orchestrator.py
"""

import asyncio
import logging
import warnings
import sys

from .models import RAGConfig, ChunkingConfig, env_settings
from .pipeline.rag_pipeline import RAGPipeline
from .utils import list_files_in_folder
from blueprints.knowledge.store import knowledge_data_raw
from blueprints.context.instruction import context_blueprints

# Suppress known asyncio cleanup warning on Windows
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio.proactor_events")

# Silence Azure HTTP logging (request/response headers)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Also silence Search-specific logs
logging.getLogger("azure.search.documents").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)

# Configure logging for visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    # Step 1: Define configuration
    config = RAGConfig(
        azure_openai_endpoint=str(env_settings.azure_endpoint_url),
        azure_openai_api_key=env_settings.azure_openai_api_key,
        azure_openai_api_version=env_settings.azure_openai_version,
        embedding_deployment=env_settings.text_embedding,
        model_deployment=env_settings.azure_deployment_name,
        azure_search_endpoint=str(env_settings.azure_ai_search_url),
        azure_search_api_key=env_settings.azure_ai_search_api_key,
        index_name=env_settings.rag_index_name,
        default_namespace=env_settings.rag_namespace_knowledge_store,
        chunking=ChunkingConfig(
            use_token_chunking=True,
            chunk_size=400,
            overlap=50,
        ),
    )

    # Step 2: Sample documents to ingest
    documents = list_files_in_folder("blueprints\sources")
    print(f"\n📄 Ingesting {len(documents)} documents from 'blueprints/sources'...")
    print("\n🗂 Sample documents:", documents[:3])
    blueprints = context_blueprints
    
    # Step 3: Use the pipeline
    async with RAGPipeline(config) as pipeline:
        if env_settings.start_with_clean_index:
            logging.info("Deleting existing index for a clean start...")
            await pipeline.index_manager.delete_index()
            logging.info("Index deleted.")

            # BUILD + INGEST blueprints
            logging.info("Uploading contextual blueprints...")
            result = await pipeline.ingest_blueprints(
                blueprints,
                namespace=env_settings.rag_namespace_blueprint_context
            )
            logging.info(f"Blueprints uploaded: {result}")

            # BUILD + INGEST knowledge
            logging.info("Uploading knowledge documents...")
            result = await pipeline.setup(
                    documents,
                    namespace=env_settings.rag_namespace_knowledge_store
            )
            logging.info(f"Knowledge uploaded: {result}")
        
        # SEARCH + ANSWER
        question = "What happened in WW2?"
        print(f"\n🔍 Question: {question}")
        answer = await pipeline.answer_question(question, top_k=3)
        print(f"💬 Answer: {answer}")

         # Direct SEARCH access
        print("\n🔬 Direct search results:")
        search_results = await pipeline.search(question, top_k=2)
        for i, result in enumerate(search_results, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result.score:.4f}")
            print(f"  Source: {result.source_id}")
            print(f"  Chunk: {result.chunk[:100]}...")

        # MULTI-AGENT CONTEXTUAL GENERATION
        large_text_from_researcher = """
        World war 1 was a global war that lasted from 1914 to 1918 involving many of the world's great powers...
        """
        goal = f"""First, summarize the following text about the World War 1 to extract only the key facts. 
Then, using that summary, write a short, technical summary of the reason and after effect of the topic.

--- TEXT TO USE ---
        {large_text_from_researcher}
        """
        print(f"\n🧠 Executing multi-agent workflow for goal:\n{goal}")
        final_output = await pipeline.generate_with_context(goal)
        print("\n🎬 Final Output:\n", final_output)

def run_main():
    if sys.platform == "win32":
        import asyncio.proactor_events

        # Patch to suppress RuntimeError when event loop is closed
        def safe_del(self):
            try:
                if self._loop.is_closed():
                    return
                self.close()
            except Exception:
                pass

        asyncio.proactor_events._ProactorBasePipeTransport.__del__ = safe_del

        # Use ProactorEventLoopPolicy explicitly for Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())

if __name__ == "__main__":
    run_main()
