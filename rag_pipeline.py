

import logging
from typing import Any, Dict, List, Optional

from models import EmbeddedChunk, SearchResult
from embedding_service import EmbeddingService
from search_service import SearchService
from llm_service import BaseLLM

logger = logging.getLogger(__name__)


class RAGPipeline:

    PROMPT_TEMPLATE = (
        "Дай ответ на данный вопрос, используя информацию из текста:\n"
        "{question}\n"
        "Текст:\n"
        "{context}"
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_service: SearchService,
        llm: BaseLLM,
        top_n: int = 5,
        top_m: int = 5,
    ):
        self.embedding_service = embedding_service
        self.search_service = search_service
        self.llm = llm
        self.top_n = top_n
        self.top_m = top_m

        self.embedded_chunks: List[EmbeddedChunk] = []

    def index_texts(
        self,
        ontology_texts: List[Dict[str, Any]],
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> int:
        logger.info("Фаза 1: индексация %d текстов ...", len(ontology_texts))

        chunks = self.embedding_service.get_chunks(
            texts=ontology_texts,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        logger.info("  Получено %d чанков.", len(chunks))

        self.embedded_chunks = self.embedding_service.embed_chunks(chunks)
        logger.info("  Эмбеддинги вычислены (%d чанков).", len(self.embedded_chunks))

        return len(self.embedded_chunks)


    def _phase2(self, question: str) -> tuple[str, List[SearchResult]]:
        logger.info("Фаза 2: поиск по вопросу ...")
        results_n = self.search_service.find_most_similar(
            query=question,
            stored_chunks=self.embedded_chunks,
            top_k=self.top_n,
        )
        logger.info("  Найдено %d чанков (N).", len(results_n))

        context = self._build_context(results_n)
        prompt = self.PROMPT_TEMPLATE.format(question=question, context=context)

        logger.info("  Отправка промпта в LLM (фаза 2) ...")
        intermediate_answer = self.llm.generate(prompt)
        logger.info("  Промежуточный ответ получен (%d симв.).", len(intermediate_answer))

        return intermediate_answer, results_n


    def _phase3(
        self,
        question: str,
        intermediate_answer: str,
        results_n: List[SearchResult],
    ) -> tuple[str, List[SearchResult]]:
        logger.info("Фаза 3: уточняющий поиск по ответу LLM ...")
        answer_emb = self.embedding_service.get_embeddings([intermediate_answer])[0]
        results_m = self.search_service.find_by_embedding(
            query_embedding=answer_emb,
            stored_chunks=self.embedded_chunks,
            top_k=self.top_m,
        )
        logger.info("  Найдено %d чанков (M).", len(results_m))

        # Объединяем N + M с дедупликацией по source_id
        combined = self._merge_results(results_n, results_m)
        logger.info("  Уникальных чанков (N+M): %d.", len(combined))

        context = self._build_context(combined)
        prompt = self.PROMPT_TEMPLATE.format(question=question, context=context)

        logger.info("  Отправка промпта в LLM (фаза 3) ...")
        final_answer = self.llm.generate(prompt)
        logger.info("  Финальный ответ получен (%d симв.).", len(final_answer))

        return final_answer, combined

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.embedded_chunks:
            raise RuntimeError(
                "Индекс пуст. Сначала вызовите index_texts() (фаза 1)."
            )

        intermediate, results_n = self._phase2(question)
        final, combined = self._phase3(question, intermediate, results_n)

        return {
            "question": question,
            "intermediate_answer": intermediate,
            "final_answer": final,
            "phase2_chunks": [
                {"score": r.score, "text": r.chunk.text, "source_id": r.source_id}
                for r in results_n
            ],
            "phase3_chunks": [
                {"score": r.score, "text": r.chunk.text, "source_id": r.source_id}
                for r in combined
            ],
        }

    @staticmethod
    def _build_context(results: List[SearchResult]) -> str:
        paragraphs = []
        for r in results:
            paragraphs.append(r.chunk.text)
        return "\n\n".join(paragraphs)

    @staticmethod
    def _merge_results(
        list_a: List[SearchResult],
        list_b: List[SearchResult],
    ) -> List[SearchResult]:
        seen: set[str] = set()
        merged: List[SearchResult] = []
        for r in list_a + list_b:
            key = r.chunk.chunk_id
            if key not in seen:
                seen.add(key)
                merged.append(r)
        return merged
