from __future__ import annotations
from typing import List, Dict, Optional, Any


class AnswerGenerator:
   
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 100,
        generator: Optional[Any] = None,
    ):
        if generator is not None:
            self._gen = generator
            self.device = getattr(generator, "device", device)
            return

        # Import here to avoid circular imports.
        from .rag_components import LlamaKVGenerator

        self._gen = LlamaKVGenerator(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        self.device = self._gen.device

    def generate_baseline(self, query: str) -> str:
       
        return self._gen.generate_fresh(query, [])

    def generate_with_rag(self, query: str, knowledge_docs: List[Dict]) -> str:
        
        return self._gen.generate_fresh(query, knowledge_docs)

    def generate_with_kv_cache(self, query: str, kv_cache, kv_len: int) -> str:
       
        if kv_cache is None or kv_len <= 0:
            return self.generate_baseline(query)
        return self._gen.generate_with_kv_cache(query, kv_cache, kv_len)

    def generate_with_cached_context(self, query: str, context_string: str) -> str:
       
        if not context_string or not context_string.strip():
            return self.generate_baseline(query)
        return self._gen.generate_fresh(
            query,
            [{"text": context_string, "id": 0, "retrieval_score": 1.0}],
        )
