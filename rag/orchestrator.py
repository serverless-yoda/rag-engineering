
"""
Main orchestrator with DI container.
"""

import asyncio
import logging
import warnings
import sys

from .models import env_settings
from .models.config import RAGConfig, ChunkingConfig
from .di.container import Container
from .utils import list_files_in_folder
from blueprints.knowledge.store import knowledge_data_raw
from blueprints.context.instruction import context_blueprints

warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio.proactor_events")

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.search.documents").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    # Step 1: Build configuration
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
        content_safety_endpoint=str(env_settings.content_safety_endpoint) if env_settings.content_safety_endpoint else None,
        content_safety_api_key=env_settings.content_safety_api_key,
        content_moderation_enabled=env_settings.content_moderation_enabled,
        content_moderation_threshold=env_settings.content_moderation_threshold,
    )
    
    # Step 2: Initialize DI container
    container = Container()
    container.config.from_dict(config.__dict__)
    
    # Step 3: Get pipeline from container
    pipeline = container.rag_pipeline()
    
    # Step 4: Sample documents
    documents = list_files_in_folder("blueprints/sources")
    blueprints = context_blueprints
    
    # Step 5: Use pipeline
    async with pipeline:
        if env_settings.start_with_clean_index:
            logging.info("🗑️  Deleting existing index...")
            # #await pipeline.index_manager.delete_index()
            
            # logging.info("📚 Uploading blueprints...")
            # result = await pipeline.ingester.ingest_blueprints(
            #     blueprints,
            #     namespace=env_settings.rag_namespace_blueprint_context
            # )
            # logging.info(f"✅ Blueprints: {result}")
            
            # logging.info("📄 Uploading documents...")
            # result = await pipeline.setup(
            #     documents,
            #     namespace=env_settings.rag_namespace_knowledge_store
            # )
            # logging.info(f"✅ Documents: {result}")
        
        # SEARCH + ANSWER
        #question = "What's the caused of  in WW1?"
        #print(f"\n🔍 Question: {question}")
        #answer = await pipeline.answer_question(question, top_k=3)
        #print(f"💬 Answer: {answer}")
        
        # Token usage
        #print(pipeline.token_tracker.report())
        
        # MULTI-AGENT WORKFLOW
        goal = """Write a short technical summary about World War 2."""
        print(f"\n🧠 Multi-Agent Goal: {goal}")
        output = await pipeline.generate_with_context(goal)
        print(f"\n🎬 Final Output:\n{output}")
        
        # Final token report
        #print(pipeline.token_tracker.report())


def run_main():
    if sys.platform == "win32":
        import asyncio.proactor_events
        
        def safe_del(self):
            try:
                if self._loop.is_closed():
                    return
                self.close()
            except Exception:
                logging.error("Error during ProactorBasePipeTransport deletion", exc_info=True)
        
        asyncio.proactor_events._ProactorBasePipeTransport.__del__ = safe_del
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())

if __name__ == "__main__":
    run_main()
