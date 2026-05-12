from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from embedding_service import EmbeddingService
from llm_service import BaseLLM
from text_json_loader import AnnotationCorpus

logger = logging.getLogger(__name__)


@dataclass
class EntityVector:
    node_uri: str
    name_ru: str
    node_type: str
    text: str
    embedding: List[float]


@dataclass
class EntityHit:
    entity: EntityVector
    score: float


@dataclass
class FragmentHit:
    text: str
    score: float
    source_entity: str


class RAGPipeline:

    PROMPT_PHASE2 = (
        "Ответь на заданный вопрос: {question}\n"
        "Используя основной текст:\n"
        "{ontology_texts}"
    )

    PROMPT_FINAL = (
        "Ответь на заданный вопрос: {question}\n"
        "Используя основной текст:\n"
        "{ontology_texts}\n"
        "Дополняя свой ответ данными текстами:\n"
        "{fragments}"
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm: BaseLLM,
        annotation_corpus: AnnotationCorpus,
        top_n: int = 5,
        top_m: int = 5,
        sentence_window_k: int = 2,
        fragments_per_entity_l: int = 3,
    ):
        self.embedding_service = embedding_service
        self.llm = llm
        self.annotation_corpus = annotation_corpus
        self.top_n = top_n
        self.top_m = top_m
        self.k = sentence_window_k
        self.l = fragments_per_entity_l
        self.entities: List[EntityVector] = []

    def index_ontology_entities(self, ontology_texts: List[Dict[str, Any]]) -> int:
        texts: List[str] = []
        meta: List[Dict[str, Any]] = []
        for item in ontology_texts:
            t = (item.get("text") or "").strip()
            if not t:
                continue
            texts.append(t)
            meta.append(item)

        if not texts:
            self.entities = []
            return 0

        embeddings = self.embedding_service.get_embeddings(texts)
        self.entities = [
            EntityVector(
                node_uri=str(m.get("source_id", "")),
                name_ru=str(m.get("name_ru", "") or ""),
                node_type=str(m.get("source_type", "")),
                text=t,
                embedding=e,
            )
            for m, t, e in zip(meta, texts, embeddings)
        ]
        logger.info("Проиндексировано сущностей онтологии: %d", len(self.entities))
        return len(self.entities)

    def _top_entities_by_embedding(
        self, query_emb: List[float], top_k: int,
    ) -> List[EntityHit]:
        hits: List[EntityHit] = []
        for ent in self.entities:
            score = self.embedding_service.cos_compare(query_emb, ent.embedding)
            hits.append(EntityHit(entity=ent, score=score))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]

    def _merge_entities(
        self, list_n: List[EntityHit], list_m: List[EntityHit],
    ) -> List[EntityHit]:
        seen: set = set()
        out: List[EntityHit] = []
        for h in list_n + list_m:
            if h.entity.node_uri in seen:
                continue
            seen.add(h.entity.node_uri)
            out.append(h)
        return out

    def _fragments_for_entity(
        self, entity: EntityVector, query_emb: List[float],
    ) -> List[FragmentHit]:
        fragments = self.annotation_corpus.fragments_for(entity.node_uri, k=self.k)
        if not fragments:
            return []

        embeddings = self.embedding_service.get_embeddings(fragments)
        scored: List[FragmentHit] = []
        for frag, emb in zip(fragments, embeddings):
            score = self.embedding_service.cos_compare(query_emb, emb)
            scored.append(
                FragmentHit(text=frag, score=score, source_entity=entity.node_uri)
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: self.l]

    def _format_ontology_block(self, entities: List[EntityHit]) -> str:
        parts: List[str] = []
        for h in entities:
            name = h.entity.name_ru or h.entity.node_uri.rsplit("/", 1)[-1]
            parts.append(f"[{name}]\n{h.entity.text}")
        return "\n\n".join(parts)

    def _format_fragments_block(
        self,
        entities: List[EntityHit],
        fragments_by_entity: Dict[str, List[FragmentHit]],
    ) -> str:
        parts: List[str] = []
        for h in entities:
            frags = fragments_by_entity.get(h.entity.node_uri, [])
            if not frags:
                continue
            name = h.entity.name_ru or h.entity.node_uri.rsplit("/", 1)[-1]
            block = [f"[{name}]"]
            for f in frags:
                block.append(f"— {f.text}")
            parts.append("\n".join(block))
        return "\n\n".join(parts)

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.entities:
            raise RuntimeError(
                "Сущности не проиндексированы. Сначала вызовите index_ontology_entities()."
            )

        logger.info("Фаза 2: поиск N сущностей по вопросу ...")
        query_emb = self.embedding_service.get_embeddings([question])[0]
        hits_n = self._top_entities_by_embedding(query_emb, self.top_n)
        logger.info("  Найдено N=%d сущностей.", len(hits_n))

        ontology_block_n = self._format_ontology_block(hits_n)
        intermediate_prompt = self.PROMPT_PHASE2.format(
            question=question, ontology_texts=ontology_block_n,
        )
        logger.info("  Отправка промпта в LLM (фаза 2) ...")
        intermediate_answer = self.llm.generate(intermediate_prompt)
        logger.info("  Промежуточный ответ получен (%d симв.).", len(intermediate_answer))

        logger.info("Фаза 3: уточняющий поиск M сущностей по ответу ...")
        answer_emb = self.embedding_service.get_embeddings([intermediate_answer])[0]
        hits_m = self._top_entities_by_embedding(answer_emb, self.top_m)
        logger.info("  Найдено M=%d сущностей.", len(hits_m))

        combined = self._merge_entities(hits_n, hits_m)
        logger.info("  Уникальных сущностей (N+M): %d.", len(combined))

        logger.info(
            "Фаза 3.5: подбор фрагментов для %d сущностей (K=%d, L=%d) ...",
            len(combined), self.k, self.l,
        )
        fragments_by_entity: Dict[str, List[FragmentHit]] = {}
        for hit in combined:
            top_frags = self._fragments_for_entity(hit.entity, query_emb)
            if top_frags:
                fragments_by_entity[hit.entity.node_uri] = top_frags

        total_frags = sum(len(v) for v in fragments_by_entity.values())
        logger.info("  Всего отобрано фрагментов: %d.", total_frags)

        ontology_block = self._format_ontology_block(combined)
        fragments_block = self._format_fragments_block(combined, fragments_by_entity)

        final_prompt = self.PROMPT_FINAL.format(
            question=question,
            ontology_texts=ontology_block,
            fragments=fragments_block if fragments_block else "(нет дополнительных фрагментов)",
        )

        logger.info("Финальный запрос к LLM ...")
        final_answer = self.llm.generate(final_prompt)
        logger.info("Финальный ответ получен (%d симв.).", len(final_answer))

        return {
            "question": question,
            "intermediate_answer": intermediate_answer,
            "final_answer": final_answer,
            "phase2_entities": [
                {
                    "name": h.entity.name_ru,
                    "uri": h.entity.node_uri,
                    "score": h.score,
                }
                for h in hits_n
            ],
            "phase3_entities": [
                {
                    "name": h.entity.name_ru,
                    "uri": h.entity.node_uri,
                    "score": h.score,
                }
                for h in hits_m
            ],
            "combined_entities": [
                {
                    "name": h.entity.name_ru,
                    "uri": h.entity.node_uri,
                    "score": h.score,
                }
                for h in combined
            ],
            "fragments": [
                {
                    "entity": h.entity.name_ru or h.entity.node_uri,
                    "text": f.text,
                    "score": f.score,
                }
                for h in combined
                for f in fragments_by_entity.get(h.entity.node_uri, [])
            ],
        }