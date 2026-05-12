from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RDF_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_INDIVIDUAL = "http://www.w3.org/2002/07/owl#NamedIndividual"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"

_SENT_END = re.compile(r"[.!?…]+\s*$")
_TRAILING_PUNCT = ".,;:!?»)]}"


@dataclass
class EntityOccurrence:
    file_id: str
    pos_start: int
    pos_end: int


@dataclass
class AnnotatedDocument:
    file_id: str
    original_text: str
    text_with_ids: Dict[int, str]
    sentence_spans: List[Tuple[int, int]] = field(default_factory=list)

    def extract_fragment(self, pos_start: int, pos_end: int, k: int = 2) -> str:
        if not self.sentence_spans:
            return ""

        matching: List[int] = []
        for i, (s, e) in enumerate(self.sentence_spans):
            if not (pos_end < s or pos_start > e):
                matching.append(i)

        if not matching:
            nearest = None
            best_dist: Optional[int] = None
            for i, (s, e) in enumerate(self.sentence_spans):
                dist = min(
                    abs(pos_start - s), abs(pos_start - e),
                    abs(pos_end - s), abs(pos_end - e),
                )
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    nearest = i
            if nearest is None:
                return ""
            matching = [nearest]

        start_idx = max(0, min(matching) - k)
        end_idx = min(len(self.sentence_spans) - 1, max(matching) + k)
        first_id = self.sentence_spans[start_idx][0]
        last_id = self.sentence_spans[end_idx][1]

        tokens: List[str] = []
        for wid in sorted(self.text_with_ids.keys()):
            if first_id <= wid <= last_id:
                token = self.text_with_ids[wid]
                if token:
                    tokens.append(token)
        return _join_tokens(tokens)


@dataclass
class AnnotationCorpus:
    documents: Dict[str, AnnotatedDocument] = field(default_factory=dict)
    occurrences: Dict[str, List[EntityOccurrence]] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    node_types: Dict[str, str] = field(default_factory=dict)

    def fragments_for(self, node_uri: str, k: int = 2) -> List[str]:
        out: List[str] = []
        seen: set = set()
        for occ in self.occurrences.get(node_uri, []):
            doc = self.documents.get(occ.file_id)
            if doc is None:
                continue
            frag = doc.extract_fragment(occ.pos_start, occ.pos_end, k=k)
            if frag and frag not in seen:
                seen.add(frag)
                out.append(frag)
        return out

    def has_mentions(self, node_uri: str) -> bool:
        return bool(self.occurrences.get(node_uri))


def _join_tokens(tokens: List[str]) -> str:
    out: List[str] = []
    for w in tokens:
        if not w:
            continue
        if not out:
            out.append(w)
        elif w[0] in _TRAILING_PUNCT:
            out.append(w)
        elif out[-1].endswith(" "):
            out.append(w)
        else:
            out.append(" " + w)
    text = "".join(out)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _ru_name(node_data: dict) -> str:
    lbl = node_data.get(
        RDF_LABEL,
        node_data.get("params_values", {}).get(RDF_LABEL, []),
    )
    if isinstance(lbl, str):
        lbl = [lbl]
    for l in lbl:
        if "@ru" in l:
            return l.replace("@ru", "").strip()
    return lbl[0].split("@")[0].strip() if lbl else ""


def _node_type(node_data: dict) -> str:
    labels = node_data.get("labels", [])
    if any(OWL_CLASS in l for l in labels):
        return "Class"
    if any(OWL_INDIVIDUAL in l for l in labels):
        return "Individual"
    return "Other"


def _build_sentence_spans(text_with_ids: Dict[int, str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    sent_start: Optional[int] = None
    last_id: Optional[int] = None
    for wid in sorted(text_with_ids.keys()):
        token = text_with_ids[wid]
        if not token or not token.strip():
            continue
        if sent_start is None:
            sent_start = wid
        last_id = wid
        if _SENT_END.search(token):
            spans.append((sent_start, wid))
            sent_start = None
    if sent_start is not None and last_id is not None:
        spans.append((sent_start, last_id))
    return spans


def find_annotation_files(directory: str | Path = ".") -> List[Path]:
    directory = Path(directory)
    files: List[Path] = []
    for p in sorted(directory.glob("*.json")):
        if p.name == "graph.json":
            continue
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        if "entites" in doc and "textWithIds" in doc:
            files.append(p)
    return files


def load_text_annotations(paths: List[str | Path]) -> AnnotationCorpus:
    corpus = AnnotationCorpus()

    for path in paths:
        path = Path(path)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue

        file_id = path.stem
        original_raw = doc.get("originalText", "")
        if isinstance(original_raw, list):
            original_text = "\n".join(str(line) for line in original_raw)
        else:
            original_text = str(original_raw)

        twi_raw = doc.get("textWithIds", {})
        text_with_ids: Dict[int, str] = {}
        for k, v in twi_raw.items():
            try:
                text_with_ids[int(k)] = str(v)
            except (ValueError, TypeError):
                continue

        spans = _build_sentence_spans(text_with_ids)
        corpus.documents[file_id] = AnnotatedDocument(
            file_id=file_id,
            original_text=original_text,
            text_with_ids=text_with_ids,
            sentence_spans=spans,
        )

        for ent in doc.get("entites", []):
            node_uri = ent.get("node_uri", "")
            if not node_uri:
                continue

            node_data = ent.get("node", {}).get("data", {})
            ntype = _node_type(node_data)
            if ntype == "Other":
                continue

            if node_uri not in corpus.names:
                name = _ru_name(node_data)
                if name:
                    corpus.names[node_uri] = name
            corpus.node_types[node_uri] = ntype

            try:
                pos_start = int(ent.get("pos_start", -1))
                pos_end = int(ent.get("pos_end", pos_start))
            except (ValueError, TypeError):
                continue
            if pos_start < 0:
                continue

            corpus.occurrences.setdefault(node_uri, []).append(
                EntityOccurrence(
                    file_id=file_id,
                    pos_start=pos_start,
                    pos_end=pos_end,
                )
            )

    return corpus