
from embedding_service import EmbeddingService  


from search_service import SearchService  


from models import TextChunk, EmbeddedChunk  

from typing import Any, Dict, List

class EmbeddingSearchService:

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._search = SearchService(embedding_service)

    def find_most_similar(
        self,
        query: str,
        stored_chunks: list,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        
        
        
        embedded = []
        for item in stored_chunks:
            if isinstance(item, EmbeddedChunk):
                embedded.append(item)
                continue

            
            if isinstance(item, dict):
                chunk = item.get("chunk")
                emb = item.get("embedding")
                if chunk is None or emb is None:
                    continue
                if isinstance(chunk, TextChunk):
                    tc = chunk
                elif isinstance(chunk, dict):
                    tc = TextChunk(**chunk)
                else:
                    tc = chunk
                embedded.append(EmbeddedChunk(chunk=tc, embedding=emb))

        results = self._search.find_most_similar(
            query=query,
            stored_chunks=embedded,
            top_k=top_k,
        )

        
        return [
            {"score": r.score, "chunk": {"chunk": r.chunk}}
            for r in results
        ]
