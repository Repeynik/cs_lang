

from typing import Any, Dict, List

from embedding_service import EmbeddingService
from ontology_repository import OntologyRepository


class OntologyEmbeddingPipeline:

    def __init__(
        self,
        ontology_repo: OntologyRepository,
        embedding_service: EmbeddingService,
    ):
        self.ontology_repo = ontology_repo
        self.embedding_service = embedding_service

    def build_embeddings_for_ontology(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        texts = self.ontology_repo.collect_ontology_texts()
        chunks = self.embedding_service.get_chunks(
            texts=texts,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        embedded = self.embedding_service.embed_chunks(chunks)
        return [
            {"chunk": ec.chunk, "embedding": ec.embedding}
            for ec in embedded
        ]

    def save_embeddings_to_neo4j(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        embedded_chunks = self.build_embeddings_for_ontology(
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for item in embedded_chunks:
            chunk = item["chunk"]
            embedding = item["embedding"]

            self.ontology_repo.save_text_chunk(
                source_uri=chunk.source_id,
                source_type=chunk.source_type,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=embedding,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )

        return embedded_chunks
