
"""
Azure AI Content Safety integration for content moderation.
"""

import logging
from typing import Dict, Any, Optional
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class ContentSafetyError(Exception):
    """Custom exception for content safety violations."""
    pass

class AzureContentSafety:
    """
    Azure AI Content Safety provider with circuit breaker protection.
    
    Monitors:
    - Hate speech
    - Self-harm
    - Sexual content
    - Violence
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        severity_threshold: int = 2,
        enabled: bool = True,
    ):
        """
        Initialize Content Safety client.
        
        Args:
            endpoint: Azure Content Safety endpoint
            api_key: API key
            severity_threshold: Block content with severity >= this (0-6)
            enabled: Enable/disable moderation
        """
        self.severity_threshold = severity_threshold
        self.enabled = enabled
        
        if enabled and endpoint and api_key:
            self.client = ContentSafetyClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
        else:
            self.client = None
            if enabled:
                logging.warning("Content Safety credentials missing. Moderation disabled.")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def moderate_text(self, text: str) -> Dict[str, Any]:
        """
        Moderate text content with retry protection.
        
        Args:
            text: Text to moderate
        
        Returns:
            Dict with moderation results
        """
        if not self.enabled or not self.client:
            return {
                "is_safe": True,
                "severity_scores": {},
                "blocked_categories": [],
                "recommendation": "Moderation disabled",
            }
        
        if not text or not text.strip():
            return {
                "is_safe": True,
                "severity_scores": {},
                "blocked_categories": [],
                "recommendation": "No content to moderate",
            }
        
        try:
            request = AnalyzeTextOptions(text=text)
            response = self.client.analyze_text(request)
            
            severity_scores = {}
            blocked_categories = []
            
            for category_result in response.categories_analysis:
                category_name = category_result.category.value
                severity = category_result.severity
                
                severity_scores[category_name] = severity
                
                if severity >= self.severity_threshold:
                    blocked_categories.append(category_name)
            
            is_safe = len(blocked_categories) == 0
            
            result = {
                "is_safe": is_safe,
                "severity_scores": severity_scores,
                "blocked_categories": blocked_categories,
                "recommendation": (
                    "✅ Content approved" if is_safe 
                    else f"⚠️ Blocked: {', '.join(blocked_categories)}"
                ),
            }
            
            if not is_safe:
                logging.warning(f"Content moderation failed: {result['recommendation']}")
                logging.debug(f"Severity scores: {severity_scores}")
            
            return result
            
        except Exception as e:
            logging.error(f"Content Safety API error: {e}")
            # Fail-open: allow content but log error
            return {
                "is_safe": True,
                "severity_scores": {},
                "blocked_categories": [],
                "recommendation": f"⚠️ Moderation service unavailable: {str(e)}",
                "error": str(e),
            }
    
    
    # implementations/azure_content_safety.py
    async def close(self) -> None:
        """
        Close the Azure Content Safety client connection.
        Safe to call multiple times.
        """
        try:
            if self.client:
                await self.client.close()
            else:
                logging.debug("AzureContentSafety client is None — nothing to close.")
        except Exception as e:
            logging.debug(f"Error closing Azure Content Safety: {e}")

    
