import re
import uuid
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models import EmbeddedChunk, TextChunk


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    
    
    

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def cos_compare(self, emb1: List[float], emb2: List[float]) -> float:
        if not emb1 or not emb2:
            raise ValueError("Эмбеддинги не должны быть пустыми")
        v1 = np.array(emb1).reshape(1, -1)
        v2 = np.array(emb2).reshape(1, -1)
        return float(cosine_similarity(v1, v2)[0][0])

    
    
    

    def get_chunks(
        self,
        texts: List[Dict[str, Any]],
        chunk_size: int = 500,
        overlap: int = 100,
        min_chunk_length: int = 30,
    ) -> List[TextChunk]:

        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap должен быть ≥ 0 и < chunk_size")

        result: List[TextChunk] = []
        for item in texts:
            source_id = str(item.get("source_id", "unknown"))
            source_type = str(item.get("source_type", "unknown"))
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            normalized = self._normalize(text)
            spans = self._split_spans(normalized, chunk_size, overlap)

            for idx, (start, end) in enumerate(spans):
                chunk_text = normalized[start:end].strip()
                if len(chunk_text) < min_chunk_length:
                    continue
                result.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        source_id=source_id,
                        source_type=source_type,
                        text=chunk_text,
                        chunk_index=idx,
                        start_char=start,
                        end_char=end,
                    )
                )
        return result

    def embed_chunks(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        if not chunks:
            return []
        texts = [c.text for c in chunks]
        embeddings = self.get_embeddings(texts)
        return [
            EmbeddedChunk(chunk=c, embedding=e)
            for c, e in zip(chunks, embeddings)
        ]

    
    
    

    def _normalize(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_spans(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[tuple[int, int]]:
        spans: List[tuple[int, int]] = []
        length = len(text)
        start = 0
        while start < length:
            raw_end = min(start + chunk_size, length)
            if raw_end < length:
                end = self._find_split(text, start, raw_end)
            else:
                end = raw_end
            spans.append((start, end))
            if end >= length:
                break
            start = max(0, end - overlap)
        return spans

    def _find_split(self, text: str, start: int, raw_end: int) -> int:
        window = max(start, raw_end - 120)
        candidate = text[window:raw_end]
        for pat in ["\n\n", ". ", "! ", "? ", "\t", "\n"]:
            pos = candidate.rfind(pat)
            if pos != -1:
                return window + pos + len(pat)
        return raw_end
