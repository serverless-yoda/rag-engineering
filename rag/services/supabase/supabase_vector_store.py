"""
🔥 FULLY ASYNC SupabaseVectorStore
Supports: ingest_blueprints(), setup(), vector_search(), get_document_count()
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from supabase import acreate_client, AsyncClient
except ImportError:
    from supabase._async.client import AsyncClient, create_client as acreate_client

from ...interfaces import VectorStoreProvider as VectorStore
from rag.models.config import RAGConfig

logger = logging.getLogger(__name__)

class SupabaseVectorStore(VectorStore):
    """🔥 NATIVE ASYNC Supabase vector store."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.table_name = config.supabase_table_name
        self._client: Optional[AsyncClient] = None
        self.is_connected = False

    async def connect(self) -> None:
        """Initialize the AsyncClient."""
        if self.is_connected and self._client:
            return
            
        try:
            self._client = await acreate_client(
                str(self.config.supabase_endpoint_url),
                self.config.supabase_service_role_key
            )
            self.is_connected = True
            logger.info("✅ Supabase AsyncClient connected")
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            raise

    async def _ensure_connected(self):
        """Internal guard to ensure client is ready before any operation."""
        if not self.is_connected or self._client is None:
            await self.connect()

    async def save_documents(self, documents: List[Dict[str, Any]], namespace: str = "default") -> Dict[str, int]:
        """Upsert documents into Supabase."""
        await self._ensure_connected()
        
        logger.info(f"📤 Saving {len(documents)} docs to namespace '{namespace}'")
        
        texts, embeddings, metadata_list = self._extract_from_documents(documents)
        valid_data = []
        
        for i in range(len(texts)):
            if embeddings[i] is not None:
                valid_data.append({
                    "content": texts[i],
                    "embedding": embeddings[i],
                    "namespace": namespace,
                    "metadata": metadata_list[i]
                })

        if not valid_data:
            return {"inserted": 0, "failed": len(documents)}
            
        # Execute Async Upsert
        result = await self._client.table(self.table_name).upsert(valid_data).execute()
        return {"inserted": len(valid_data), "failed": 0}

    async def vector_search(self, query_vector: List[float], top_k: int = 5, 
                           filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search via RPC."""
        await self._ensure_connected()
        
        # Simple extraction of namespace from filter string if provided
        namespace = None
        if filter_expr and "namespace eq '" in filter_expr:
            namespace = filter_expr.split("namespace eq '")[-1].split("'")[0]
        
        response = await self._client.rpc(
            "match_rag_vectors",
            {
                "query_embedding": query_vector, 
                "match_threshold": 0.5, 
                "match_count": top_k, 
                "namespace_param": namespace
            }
        ).execute()
        
        return [
            {
                "content": row["content"], 
                "similarity": row["similarity"], 
                "metadata": row["metadata"]
            }
            for row in response.data
        ]

    async def get_document_count(self, namespace: Optional[str] = None) -> int:
        await self._ensure_connected()
        query = self._client.table(self.table_name).select("count", count="exact")
        if namespace:
            query = query.eq("namespace", namespace)
        
        result = await query.execute()
        return result.count or 0

    async def list_documents(self, namespace: str, limit: int = 5) -> List[Dict[str, Any]]:
        await self._ensure_connected()
        result = await self._client.table(self.table_name)\
            .select("*").eq("namespace", namespace).limit(limit).execute()
        return result.data

    def _extract_from_documents(self, documents: List[Dict[str, Any]]):
        texts, embeddings, metadata_list = [], [], []
        for doc in documents:
            text = doc.get("chunk") or doc.get("page_content", "") or doc.get("content", "")
            # Support multiple embedding key names
            embedding = doc.get("chunk_vector") or doc.get("embedding")
            metadata = doc.get("metadata_json") or doc.get("metadata") or {}
            
            texts.append(text)
            embeddings.append(embedding)
            metadata_list.append(metadata)
        return texts, embeddings, metadata_list

    @property
    def client(self) -> AsyncClient:
        """Public access to the internal client (throws error if not connected)."""
        if not self._client:
            raise RuntimeError("Supabase client not connected. Call await connect() first.")
        return self._client
    
    async def close(self) -> None:
        """
        🔥 REQUIRED: Implementation of abstract method 'close'
        Clean up the Supabase AsyncClient resources.
        """
        if self._client:
            # Note: supabase-py AsyncClient uses httpx internally.
            # We call aclose() if available to shut down the connection pool.
            try:
                if hasattr(self._client, "aclose"):
                    await self._client.aclose()
                elif hasattr(self._client, "close"):
                    await self._client.close()
                
                self._client = None
                self.is_connected = False
                logger.info("🔌 Supabase AsyncClient connection pool closed.")
            except Exception as e:
                logger.warning(f"⚠️ Error during Supabase shutdown: {e}")