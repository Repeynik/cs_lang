

from typing import List

from models import EmbeddedChunk, SearchResult
from embedding_service import EmbeddingService


class SearchService:

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def find_most_similar(
        self,
        query: str,
        stored_chunks: List[EmbeddedChunk],
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not query.strip():
            return []

        query_emb = self.embedding_service.get_embeddings([query])[0]
        return self.find_by_embedding(query_emb, stored_chunks, top_k)

    def find_by_embedding(
        self,
        query_embedding: List[float],
        stored_chunks: List[EmbeddedChunk],
        top_k: int = 5,
    ) -> List[SearchResult]:
        scored: List[SearchResult] = []
        for ec in stored_chunks:
            score = self.embedding_service.cos_compare(query_embedding, ec.embedding)
            scored.append(
                SearchResult(
                    score=score,
                    chunk=ec.chunk,
                    source_id=ec.chunk.source_id,
                    source_type=ec.chunk.source_type,
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
