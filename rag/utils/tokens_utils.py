# utils/tokens_utils.py

"""
Enhanced token tracking utilities with comprehensive monitoring.
"""

import tiktoken
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from contextlib import contextmanager
import threading
import time

@dataclass
class TokenUsage:
    """Token usage metrics for operations."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
    
    def to_dict(self) -> Dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.total_tokens,
        }

class TokenTracker:
    """Thread-safe token usage tracker with stage breakdown."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._usage = TokenUsage()
        self._by_stage: Dict[str, TokenUsage] = {}
        self._start_time = time.time()
    
    def add_embedding_usage(self, texts: List[str], stage: str = "embedding"):
        """Track embedding token usage."""
        tokens = sum(count_tokens(text) for text in texts)
        with self._lock:
            self._usage.embedding_tokens += tokens
            self._usage.total_tokens += tokens
            if stage not in self._by_stage:
                self._by_stage[stage] = TokenUsage()
            self._by_stage[stage].embedding_tokens += tokens
            self._by_stage[stage].total_tokens += tokens
    
    def add_llm_usage(self, prompt_tokens: int, completion_tokens: int, stage: str = "generation"):
        """Track LLM token usage from API response."""
        with self._lock:
            self._usage.prompt_tokens += prompt_tokens
            self._usage.completion_tokens += completion_tokens
            self._usage.total_tokens += prompt_tokens + completion_tokens
            if stage not in self._by_stage:
                self._by_stage[stage] = TokenUsage()
            self._by_stage[stage].prompt_tokens += prompt_tokens
            self._by_stage[stage].completion_tokens += completion_tokens
            self._by_stage[stage].total_tokens += prompt_tokens + completion_tokens
    
    def get_usage(self) -> TokenUsage:
        """Get total usage."""
        with self._lock:
            return TokenUsage(**self._usage.to_dict())
    
    def get_stage_usage(self, stage: str) -> Optional[TokenUsage]:
        """Get usage for a specific stage."""
        with self._lock:
            usage = self._by_stage.get(stage)
            return TokenUsage(**usage.to_dict()) if usage else None
    
    def get_all_stages(self) -> Dict[str, TokenUsage]:
        """Get usage breakdown by stage."""
        with self._lock:
            return {k: TokenUsage(**v.to_dict()) for k, v in self._by_stage.items()}
    
    def reset(self):
        """Reset all counters."""
        with self._lock:
            self._usage = TokenUsage()
            self._by_stage = {}
            self._start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since tracker was created or reset."""
        return time.time() - self._start_time
    
    def estimate_cost(self, 
                      embedding_cost_per_1k: float = 0.0001,
                      prompt_cost_per_1k: float = 0.0030,
                      completion_cost_per_1k: float = 0.0060) -> Dict[str, float]:
        """Estimate API costs based on token usage."""
        with self._lock:
            embedding_cost = (self._usage.embedding_tokens / 1000) * embedding_cost_per_1k
            prompt_cost = (self._usage.prompt_tokens / 1000) * prompt_cost_per_1k
            completion_cost = (self._usage.completion_tokens / 1000) * completion_cost_per_1k
            return {
                "embedding_cost": embedding_cost,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": embedding_cost + prompt_cost + completion_cost,
            }
    
    def report(self) -> str:
        """Generate comprehensive usage report."""
        with self._lock:
            costs = self.estimate_cost()
            elapsed = self.get_elapsed_time()
            
            lines = [
                "\n" + "=" * 60,
                "📊 TOKEN USAGE REPORT",
                "=" * 60,
                f"⏱️  Elapsed Time: {elapsed:.2f}s",
                "",
                "🔢 Total Tokens:",
                f"   Total: {self._usage.total_tokens:,}",
                f"   ├─ Prompt: {self._usage.prompt_tokens:,}",
                f"   ├─ Completion: {self._usage.completion_tokens:,}",
                f"   └─ Embedding: {self._usage.embedding_tokens:,}",
                "",
                "💰 Estimated Cost (GPT-4 rates):",
                f"   Total: ${costs['total_cost']:.4f}",
                f"   ├─ Embeddings: ${costs['embedding_cost']:.4f}",
                f"   ├─ Prompts: ${costs['prompt_cost']:.4f}",
                f"   └─ Completions: ${costs['completion_cost']:.4f}",
            ]
            
            if self._by_stage:
                lines.extend([
                    "",
                    "📋 Breakdown by Stage:",
                    "-" * 60,
                ])
                for stage, usage in sorted(self._by_stage.items()):
                    lines.append(f"  📌 {stage}:")
                    lines.append(f"     Total: {usage.total_tokens:,}")
                    if usage.prompt_tokens > 0:
                        lines.append(f"     ├─ Prompt: {usage.prompt_tokens:,}")
                    if usage.completion_tokens > 0:
                        lines.append(f"     ├─ Completion: {usage.completion_tokens:,}")
                    if usage.embedding_tokens > 0:
                        lines.append(f"     └─ Embeddings: {usage.embedding_tokens:,}")
            
            lines.append("=" * 60)
            return "\n".join(lines)

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken for Azure/OpenAI models."""
    model_map = {
        "gpt-4": "gpt-4",
        "gpt-4-nano": "gpt-4",
        "gpt-35-turbo": "gpt-3.5-turbo",
        "text-embedding-ada-002": "text-embedding-ada-002",
    }
    
    base_model = model_map.get(model, model)
    try:
        encoding = tiktoken.encoding_for_model(base_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
