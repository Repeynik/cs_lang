
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import OntologyArc, OntologyGraph, OntologyNode, OntologyProperty

RDF_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_INDIVIDUAL = "http://www.w3.org/2002/07/owl#NamedIndividual"


def _extract_name(labels_list: List[str], lang: str = "ru") -> str:
    suffix = f"@{lang}"
    for item in labels_list:
        if item.endswith(suffix):
            return item[: -len(suffix)].strip()
    if labels_list:
        return labels_list[0].split("@")[0].strip()
    return ""


def _extract_names(labels_list: List[str]) -> tuple[str, str]:
    return _extract_name(labels_list, "ru"), _extract_name(labels_list, "en")

def load_ontology_from_json(path: str | Path) -> OntologyGraph:

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    raw_nodes: List[dict] = raw.get("nodes", [])
    raw_arcs: List[dict] = raw.get("arcs", [])

    graph = OntologyGraph()
    node_uri_set: set[str] = set()

    for rn in raw_nodes:
        d: dict = rn.get("data", {})
        uri: str = d.get("uri", rn.get("id", ""))
        labels: List[str] = d.get("labels", [])
        pv: dict = d.get("params_values", {})

        name_labels: List[str] = pv.get(RDF_LABEL, d.get(RDF_LABEL, []))
        if isinstance(name_labels, str):
            name_labels = [name_labels]
        name_ru, name_en = _extract_names(name_labels)

        description: str = str(pv.get(RDF_COMMENT, ""))

        is_class = any(OWL_CLASS in l for l in labels)
        is_individual = any(OWL_INDIVIDUAL in l for l in labels)
        node_type = "Class" if is_class else ("Individual" if is_individual else "Other")

        node = OntologyNode(
            uri=uri,
            name_ru=name_ru,
            name_en=name_en,
            description=description,
            node_type=node_type,
        )
        graph.nodes.append(node)
        node_uri_set.add(uri)

    properties_map: Dict[str, OntologyProperty] = {}

    for ra in raw_arcs:
        ad: dict = ra.get("data", {})
        rel_uri: str = ad.get("uri", "")
        rel_short: str = rel_uri.split("#")[-1] if "#" in rel_uri else rel_uri.split("/")[-1]
        source_uri: str = ra.get("source", "")
        target_uri: str = ra.get("target", "")

        graph.arcs.append(
            OntologyArc(
                source_uri=source_uri,
                target_uri=target_uri,
                relation_type=rel_short,
            )
        )

        if rel_short in ("domain", "range"):
            sn_data: dict = ad.get("start_node", {}).get("data", {})
            prop_uri = source_uri
            prop_labels_raw = sn_data.get("labels", [])

            is_dtp = any("DatatypeProperty" in l for l in prop_labels_raw)
            is_op = any("ObjectProperty" in l for l in prop_labels_raw)
            kind = "DatatypeProperty" if is_dtp else ("ObjectProperty" if is_op else "unknown")

            if prop_uri not in properties_map:
                pv_prop = sn_data.get("params_values", {})
                lbl_prop = pv_prop.get(RDF_LABEL, sn_data.get(RDF_LABEL, []))
                if isinstance(lbl_prop, str):
                    lbl_prop = [lbl_prop]
                pn_ru, pn_en = _extract_names(lbl_prop)
                properties_map[prop_uri] = OntologyProperty(
                    uri=prop_uri,
                    name_ru=pn_ru,
                    name_en=pn_en,
                    kind=kind,
                )

            if rel_short == "domain":
                properties_map[prop_uri].domain_class_uri = target_uri
            elif rel_short == "range":
                properties_map[prop_uri].range_class_uri = target_uri

    graph.properties = list(properties_map.values())

    for arc in graph.arcs:
        if arc.relation_type == "type":
            for node in graph.nodes:
                if node.uri == arc.source_uri and node.node_type == "Individual":
                    node.class_uri = arc.target_uri
                    break

    return graph



def build_ontology_texts(graph: OntologyGraph) -> List[Dict[str, Any]]:
    node_by_uri: Dict[str, OntologyNode] = {n.uri: n for n in graph.nodes}
    props_by_domain: Dict[str, List[OntologyProperty]] = {}
    for p in graph.properties:
        props_by_domain.setdefault(p.domain_class_uri, []).append(p)

    subclass_map: Dict[str, List[str]] = {}
    type_map: Dict[str, str] = {}

    for arc in graph.arcs:
        if arc.relation_type == "subClassOf":
            subclass_map.setdefault(arc.source_uri, []).append(arc.target_uri)
        elif arc.relation_type == "type":
            type_map[arc.source_uri] = arc.target_uri

    texts: List[Dict[str, Any]] = []

    for node in graph.nodes:
        if node.node_type not in ("Class", "Individual"):
            continue

        parts: List[str] = []

        if node.node_type == "Class":
            parts.append(f"Класс: {node.name_ru}")
            if node.description and node.description != "Описание":
                parts.append(f"Описание: {node.description}")

            parents = subclass_map.get(node.uri, [])
            for pu in parents:
                parent = node_by_uri.get(pu)
                if parent:
                    parts.append(f"Является подклассом: {parent.name_ru}")

            children = [
                node_by_uri[a.source_uri].name_ru
                for a in graph.arcs
                if a.relation_type == "subClassOf"
                and a.target_uri == node.uri
                and a.source_uri in node_by_uri
            ]
            if children:
                parts.append(f"Подклассы: {', '.join(children)}")

            instances = [
                node_by_uri[uri].name_ru
                for uri, cls_uri in type_map.items()
                if cls_uri == node.uri and uri in node_by_uri
            ]
            if instances:
                parts.append(f"Экземпляры: {', '.join(instances)}")

            class_props = props_by_domain.get(node.uri, [])
            dtp_names = [p.name_ru for p in class_props if p.kind == "DatatypeProperty"]
            op_items = [p for p in class_props if p.kind == "ObjectProperty"]

            if dtp_names:
                parts.append(f"Атрибуты: {', '.join(dtp_names)}")
            for op in op_items:
                range_node = node_by_uri.get(op.range_class_uri or "")
                range_name = range_node.name_ru if range_node else "?"
                parts.append(f"Связь «{op.name_ru}» → {range_name}")

        elif node.node_type == "Individual":
            parts.append(f"Экземпляр: {node.name_ru}")
            if node.description and node.description != "Описание":
                parts.append(f"Описание: {node.description}")

            cls_uri = type_map.get(node.uri)
            cls_node = node_by_uri.get(cls_uri or "")
            if cls_node:
                parts.append(f"Имеет тип: {cls_node.name_ru}")

                class_props = props_by_domain.get(cls_uri or "", [])
                dtp_names = [p.name_ru for p in class_props if p.kind == "DatatypeProperty"]
                if dtp_names:
                    parts.append(f"Атрибуты класса: {', '.join(dtp_names)}")

        text = ". ".join(parts)
        texts.append({
            "source_id": node.uri,
            "source_type": node.node_type,
            "text": text,
        })

    return texts
