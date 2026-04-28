
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class OntologyNode(BaseModel):
    uri: str
    name_ru: str = ""
    name_en: str = ""
    description: str = ""
    node_type: str = "" 
    class_uri: Optional[str] = None  


class OntologyProperty(BaseModel):
    uri: str
    name_ru: str = ""
    name_en: str = ""
    kind: str = ""
    domain_class_uri: str = ""
    range_class_uri: Optional[str] = None


class OntologyArc(BaseModel):
    source_uri: str
    target_uri: str
    relation_type: str = ""


class OntologyGraph(BaseModel):
    nodes: List[OntologyNode] = []
    properties: List[OntologyProperty] = []
    arcs: List[OntologyArc] = []


class TextChunk(BaseModel):
    chunk_id: str = ""
    source_id: str = ""
    source_type: str = ""
    text: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0

    def __init__(self, **data):
        super().__init__(**data)
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())


class EmbeddedChunk(BaseModel):
    chunk: TextChunk
    embedding: List[float]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class SearchResult(BaseModel):
    score: float
    chunk: TextChunk
    source_id: str = ""
    source_type: str = ""
