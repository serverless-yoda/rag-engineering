"""
Main orchestrator with DI container.
Dynamic provider support: Azure, OpenRouter, Gemini + Supabase testing.
"""

import asyncio
import logging
import warnings
import sys
import argparse
from pathlib import Path

from rag.services.supabase.supabase_vector_store import SupabaseVectorStore

from .models import env_settings
from .models.config import RAGConfig, ChunkingConfig
from .di.container import Container
from .utils.file import list_files_in_folder
from blueprints.knowledge.store import knowledge_data_raw
from blueprints.context.instruction import context_blueprints
from .models import setup_json_logging
from dependency_injector import providers

setup_json_logging("logs/pipeline.log")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio.proactor_events")

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.search.documents").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)


def parse_cli_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description="RAG Pipeline with Supabase Testing")
    parser.add_argument(
        "--test-supabase",
        action="store_true",
        help="Run Supabase upload + query test"
    )
    parser.add_argument(
        "--test-azure", 
        action="store_true",
        help="Run Azure upload + query test"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload test data (skip query)"
    )
    parser.add_argument(
        "--namespace",
        default="test_ww2",
        help="Namespace for test data (default: test_ww2)"
    )
    parser.add_argument(
        "--knowledge",
        default=None,
        help="Namespace for knowledge data (default: test_ww2)"
    )

    return parser.parse_args()


async def load_knowledge_data(pipeline):
    documents = list_files_in_folder("blueprints/sources")
    blueprints = context_blueprints
    
    logging.info("📚 Uploading blueprints...")
    result = await pipeline.ingester.ingest_blueprints(
                blueprints,
                namespace=env_settings.rag_namespace_blueprint_context
            )
    logging.info(f"✅ Blueprints: {result}")
            
    logging.info("📄 Uploading documents...")
    result = await pipeline.setup(
                documents,
                namespace=env_settings.rag_namespace_knowledge_store
            )
    logging.info(f"✅ Documents: {result}")


async def test_supabase_upload_query(pipeline, namespace: str = "test_ww2"):
    test_docs = [
        {"page_content": "WWII began 1939 Germany Poland.", "metadata": {}},
        {"page_content": "D-Day June 6 1944 Normandy.", "metadata": {}},
    ]
    
    result = await pipeline.ingester.ingest_documents(test_docs, namespace=namespace)
    logging.info(f"✅ Upload result: {result}")

    # Direct Supabase client query
    client = pipeline.store.client
    result = client.table("rag_vectors").select("*").limit(5).execute()
    print(f"✅ RAW QUERY: {len(result.data)} rows")
    for row in result.data:
        print(f"  📄 '{row['content'][:50]}...' | ns='{row['namespace']}'")


async def verify_supabase_save(pipeline, namespace="test_ww2"):
    """🔍 VERIFICATION: Check if documents REALLY saved."""
    print("\n🔍 === SUPABASE VERIFICATION ===")
    
    # Check 1: Direct count
    count = await pipeline.store.get_document_count(namespace)
    print(f"📊 Direct count: {count}")
    
    # Check 2: List documents
    docs = await pipeline.store.list_documents(namespace)
    print(f"📋 Found docs: {len(docs)}")
    for i, doc in enumerate(docs[:3]):
        print(f"   {i+1}: '{doc.get('content', 'N/A')[:50]}...' | ns='{doc.get('namespace')}'")
    
    print("🔍 === END VERIFICATION ===\n")


async def test_azure_upload_query(pipeline, namespace: str = "test_ww2"):
    """Test Azure AI Search: upload → vectorize → query."""
    test_docs = [
        "World War II (1939-1945) started with Germany's invasion of Poland.",
        "D-Day was June 6, 1944 - Allied invasion of Normandy, France.",
        "Atomic bombs dropped: Hiroshima (Aug 6, 1945), Nagasaki (Aug 9, 1945)."
    ]
    
    logging.info(f"📤 Uploading {len(test_docs)} docs to Azure namespace '{namespace}'...")
    
    result = await pipeline.ingester.ingest_documents(
        test_docs,
        namespace=namespace
    )
    logging.info(f"✅ Azure upload result: {result}")
    
    if not args.upload_only:
        question = "Tell me about D-Day in WWII"
        print(f"\n🔍 Question: {question}")
        answer = await pipeline.answer_question(question, top_k=3, namespace=namespace)
        print(f"💬 Answer: {answer}")
    
    return result


async def main():
    # Parse CLI args
    global args
    args = parse_cli_args()
    
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
        
        # Provider configuration from .env
        provider_stack=env_settings.provide_stack,
        chat_provider_type=env_settings.chat_provider_type,
        openrouter_api_key=env_settings.openrouter_api_key,
        openrouter_model=env_settings.openrouter_model,
        gemini_api_key=env_settings.gemini_api_key,
        gemini_model=env_settings.gemini_model,
        
        # Vector backend config
        vector_backend=env_settings.vector_backend,
        supabase_endpoint_url=str(env_settings.supabase_endpoint_url),
        supabase_service_role_key=env_settings.supabase_service_role_key,
        supabase_table_name=env_settings.supabase_table_name,
        supabase_vector_dimension=env_settings.supabase_vector_dimension,
        
        # Content safety
        content_safety_endpoint=str(env_settings.content_safety_endpoint) if env_settings.content_safety_endpoint else None,
        content_safety_api_key=env_settings.content_safety_api_key,
        content_moderation_enabled=env_settings.content_moderation_enabled,
        content_moderation_threshold=env_settings.content_moderation_threshold,
        
        # Pipeline defaults
        llm_timeout=60.0,
        llm_retries=3,
        batch_size=10,
        vector_dimensions=1536,
    )
    
    # Log provider selection
    logging.info(f"🚀 Vector: {config.vector_backend} | Embed: {config.provider_stack} | Chat: {config.chat_provider_type}")
    logging.info(f"📱 Model: {config.openrouter_model if config.chat_provider_type == 'openrouter' else config.model_deployment}")
    
    # Step 2: Initialize DI container
    container = Container()
    
    # 🔥 FIX: SUPABASE OVERRIDE - Pass REAL RAGConfig object
    if config.vector_backend == "supabase":
        container.supabase_store.override(providers.Factory(
            SupabaseVectorStore,
            config=config  # 🔥 Your perfect RAGConfig object
        ))
    
    # Safe config load (skip problematic Supabase fields)
    safe_config = {k: v for k, v in config.__dict__.items() 
                   if k not in ['supabase_endpoint_url', 'supabase_service_role_key']}
    container.config.from_dict(safe_config)
    
    print(f"🔍 vector_backend={config.vector_backend}")
    
    # Step 3: Get pipeline
    pipeline = container.rag_pipeline()
    
    # 🔥 CRITICAL: Initialize Supabase async client
    if config.vector_backend == "supabase" and hasattr(pipeline.store, "connect"):
        print("🔌 Connecting Supabase...")
        await pipeline.store.connect()
        print(f"✅ Store initialized: {type(pipeline.store).__name__}")

    async with pipeline:
        # SUPABASE TESTING
        if args.test_supabase:
            logging.info("🧪 Running Supabase test...")
            await test_supabase_upload_query(pipeline, args.namespace)
            await verify_supabase_save(pipeline, args.namespace)
        
        # AZURE TESTING  
        elif args.test_azure:
            logging.info("🧪 Running Azure test...")
            await test_azure_upload_query(pipeline, args.namespace)

        # LOAD KNOWLEDGE DATA
        elif args.knowledge:
            logging.info("📥 Loading knowledge data...")
            await load_knowledge_data(pipeline)

            await verify_supabase_save(pipeline, namespace=env_settings.rag_namespace_blueprint_context)
            await verify_supabase_save(pipeline, namespace=env_settings.rag_namespace_knowledge_store)
            
        # DEFAULT: Original multi-agent workflow
        else:           
            print("🤖 Starting multi-agent RAG pipeline...")
            if env_settings.start_with_clean_index:
                logging.info("🗑️  Deleting existing index...")
                # await pipeline.index_manager.delete_index()
            
            goal = "Write a short technical summary about Adolf Hitler"
            print(f"\n🧠 Multi-Agent Goal: {goal}")
            output = await pipeline.generate_with_context(goal)
            print(f"\n🎬 Final Output:\n{output}")
            print(f"\n📊 Token Usage:\n{pipeline.token_tracker.report()}")
        
        # Save execution trace
        if hasattr(pipeline, 'context_engine'):
            with open("execution_trace.log", "w", encoding="utf-8") as f:
                f.write(pipeline.context_engine.get_execution_report())
        
        logging.info("✅ Pipeline completed successfully!")


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
