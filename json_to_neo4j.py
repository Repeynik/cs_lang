from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neo4j_repository import Neo4JData, Neo4jRepository
from text_json_loader import find_annotation_files, load_text_annotations

logger = logging.getLogger(__name__)

CLASS_LABEL  = "Class"
OBJECT_LABEL = "Object"
DTP_LABEL    = "DatatypeProperty"
OP_LABEL     = "ObjectProperty"

REL_SUBCLASS_OF = "SUBCLASS_OF"
REL_HAS_DTP     = "HAS_DATATYPE_PROPERTY"
REL_HAS_OP      = "HAS_OBJECT_PROPERTY"
REL_INSTANCE_OF = "INSTANCE_OF"
REL_OP_RANGE    = "RANGE"
REL_TEXT_REL    = "TEXT_RELATION"

RDF_LABEL_KEY = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_CLASS     = "http://www.w3.org/2002/07/owl#Class"
OWL_INDIVIDUAL = "http://www.w3.org/2002/07/owl#NamedIndividual"


def _ru_name(labels_list: List[str]) -> str:
    for item in labels_list:
        if item.endswith("@ru"):
            return item[:-3].strip()
    if labels_list:
        return labels_list[0].split("@")[0].strip()
    return ""


def _parse_nodes(raw_nodes: List[dict]) -> Tuple[List[dict], List[dict]]:
    classes, individuals = [], []
    for rn in raw_nodes:
        d = rn.get("data", {})
        labels = d.get("labels", [])
        pv = d.get("params_values", {})
        uri = d.get("uri", rn.get("id", ""))

        name_labels = pv.get(RDF_LABEL_KEY, d.get(RDF_LABEL_KEY, []))
        if isinstance(name_labels, str):
            name_labels = [name_labels]

        description = str(pv.get("http://www.w3.org/2000/01/rdf-schema#comment", ""))
        if description == "Описание":
            description = ""

        info = {"uri": uri, "title": _ru_name(name_labels), "description": description}

        if any(OWL_CLASS in l for l in labels):
            classes.append(info)
        elif any(OWL_INDIVIDUAL in l for l in labels):
            individuals.append(info)

    return classes, individuals


def _parse_properties(raw_arcs: List[dict]) -> Dict[str, dict]:
    props: Dict[str, dict] = {}
    for arc in raw_arcs:
        ad = arc.get("data", {})
        rel_type = ad.get("uri", "").split("#")[-1]

        if rel_type == "domain":
            prop_uri = arc["source"]
            sn = ad.get("start_node", {}).get("data", {})
            slabels = sn.get("labels", [])
            pv = sn.get("params_values", {})
            name_labels = pv.get(RDF_LABEL_KEY, sn.get(RDF_LABEL_KEY, []))
            if isinstance(name_labels, str):
                name_labels = [name_labels]
            is_dtp = any("DatatypeProperty" in l for l in slabels)
            if prop_uri not in props:
                props[prop_uri] = {
                    "uri": prop_uri,
                    "name": _ru_name(name_labels),
                    "kind": "DatatypeProperty" if is_dtp else "ObjectProperty",
                    "domain_class_uri": arc["target"],
                    "range_class_uri": None,
                }
        elif rel_type == "range":
            prop_uri = arc["source"]
            if prop_uri in props:
                props[prop_uri]["range_class_uri"] = arc["target"]

    return props


def _parse_arcs(raw_arcs: List[dict]) -> Tuple[List[Tuple], List[Tuple]]:
    subclass_arcs, type_arcs = [], []
    for arc in raw_arcs:
        rel_type = arc["data"].get("uri", "").split("#")[-1]
        if rel_type == "subClassOf":
            subclass_arcs.append((arc["source"], arc["target"]))
        elif rel_type == "type":
            type_arcs.append((arc["source"], arc["target"]))
    return subclass_arcs, type_arcs


def _create_node_raw(repo: Neo4jRepository, label: str, props: Dict[str, Any]) -> str:
    uri = Neo4JData._generate_random_string()
    all_props = {"uri": uri, **props}
    props_str = Neo4JData._transform_props(all_props)
    rows, _ = repo.execute_query(f"CREATE (n:`{label}` {props_str}) RETURN n")
    if not rows:
        raise RuntimeError(f"Failed to create {label}")
    return uri


def _create_rel(repo: Neo4jRepository, uri1: str, uri2: str, rel: str,
                extra_props: Optional[Dict] = None) -> None:
    rel_uri = Neo4JData._generate_random_string()
    props = {"uri": rel_uri, **(extra_props or {})}
    props_str = Neo4JData._transform_props(props)
    repo.execute_query(
        f"""MATCH (a {{uri:$u1}}),(b {{uri:$u2}})
            MERGE (a)-[r:`{rel}` {props_str}]->(b)""",
        {"u1": uri1, "u2": uri2},
    )

def import_graph_json(
    json_path: str | Path,
    repo: Neo4jRepository,
    clear_first: bool = False,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    if clear_first:
        logger.warning("Clearing Neo4j before import...")
        repo.execute_query("MATCH (n) DETACH DELETE n", {})

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    raw_nodes = raw.get("nodes", [])
    raw_arcs  = raw.get("arcs",  [])

    classes, individuals = _parse_nodes(raw_nodes)
    properties  = _parse_properties(raw_arcs)
    subclass_arcs, type_arcs = _parse_arcs(raw_arcs)

    stats: Dict[str, int] = {k: 0 for k in [
        "classes", "individuals", "datatype_props", "object_props",
        "subclass_rels", "type_rels", "dtp_rels", "op_rels", "range_rels",
    ]}
    uri_map: Dict[str, str] = {}

    logger.info("Importing %d classes...", len(classes))
    for cls in classes:
        neo_uri = _create_node_raw(repo, CLASS_LABEL,
                                   {"title": cls["title"], "description": cls["description"]})
        uri_map[cls["uri"]] = neo_uri
        stats["classes"] += 1

    for child_uri, parent_uri in subclass_arcs:
        c, p = uri_map.get(child_uri), uri_map.get(parent_uri)
        if c and p:
            _create_rel(repo, c, p, REL_SUBCLASS_OF)
            stats["subclass_rels"] += 1

    dtp_list = [(u, p) for u, p in properties.items() if p["kind"] == "DatatypeProperty"]
    logger.info("Importing %d DatatypeProperties...", len(dtp_list))
    for orig_uri, prop in dtp_list:
        neo_uri = _create_node_raw(repo, DTP_LABEL,
                                   {"attr_name": prop["name"], "datatype": "string", "description": ""})
        uri_map[orig_uri] = neo_uri
        stats["datatype_props"] += 1
        domain_neo = uri_map.get(prop["domain_class_uri"])
        if domain_neo:
            _create_rel(repo, domain_neo, neo_uri, REL_HAS_DTP)
            stats["dtp_rels"] += 1

    op_list = [(u, p) for u, p in properties.items() if p["kind"] == "ObjectProperty"]
    logger.info("Importing %d ObjectProperties...", len(op_list))
    for orig_uri, prop in op_list:
        neo_uri = _create_node_raw(repo, OP_LABEL,
                                   {"attr_name": prop["name"], "description": ""})
        uri_map[orig_uri] = neo_uri
        stats["object_props"] += 1
        domain_neo = uri_map.get(prop["domain_class_uri"])
        if domain_neo:
            _create_rel(repo, domain_neo, neo_uri, REL_HAS_OP)
            stats["op_rels"] += 1
        if prop["range_class_uri"]:
            range_neo = uri_map.get(prop["range_class_uri"])
            if range_neo:
                _create_rel(repo, neo_uri, range_neo, REL_OP_RANGE)
                stats["range_rels"] += 1

    logger.info("Importing %d individuals...", len(individuals))
    for ind in individuals:
        neo_uri = _create_node_raw(repo, OBJECT_LABEL,
                                   {"title": ind["title"], "description": ind["description"]})
        uri_map[ind["uri"]] = neo_uri
        stats["individuals"] += 1

    for ind_uri, cls_uri in type_arcs:
        ind_neo = uri_map.get(ind_uri)
        cls_neo = uri_map.get(cls_uri)
        if ind_neo and cls_neo:
            _create_rel(repo, ind_neo, cls_neo, REL_INSTANCE_OF)
            stats["type_rels"] += 1

    logger.info(
        "graph.json import done: %d classes, %d individuals, %d properties",
        stats["classes"], stats["individuals"],
        stats["datatype_props"] + stats["object_props"],
    )
    return stats, uri_map

def import_text_relations(
    annotation_paths: List[str | Path],
    repo: Neo4jRepository,
    uri_map: Dict[str, str],
) -> Dict[str, int]:
    from text_json_loader import (
        load_text_annotations, _node_type, _ru_name, OWL_INDIVIDUAL
    )

    annotations = load_text_annotations(annotation_paths)
    stats = {"text_relations": 0, "new_nodes": 0}

    for node_uri, tnd in annotations.items():
        if node_uri in uri_map:
            continue
        if tnd.node_type == "Individual":
            neo_uri = _create_node_raw(repo, OBJECT_LABEL,
                                       {"title": tnd.name_ru, "description": ""})
        elif tnd.node_type == "Class":
            neo_uri = _create_node_raw(repo, CLASS_LABEL,
                                       {"title": tnd.name_ru, "description": ""})
        else:
            continue
        uri_map[node_uri] = neo_uri
        stats["new_nodes"] += 1

    for node_uri, tnd in annotations.items():
        subj_neo = uri_map.get(node_uri)
        if not subj_neo:
            continue
        for pred_name, obj_name, direction in tnd.relations:
            if direction != "out":
                continue
            q = f"""
            MATCH (o:`{OBJECT_LABEL}` {{title:$title}})
            RETURN o.uri as uri LIMIT 1
            """
            rows, _ = repo.execute_query(q, {"title": obj_name})
            if not rows:
                continue
            obj_neo = rows[0].get("uri")
            if not obj_neo or obj_neo == subj_neo:
                continue
            _create_rel(repo, subj_neo, obj_neo, REL_TEXT_REL,
                        {"predicate": pred_name})
            stats["text_relations"] += 1

    logger.info(
        "Text relations import done: %d relations, %d new nodes",
        stats["text_relations"], stats["new_nodes"],
    )
    return stats


def import_all(
    graph_json_path: str | Path,
    annotation_dir:  str | Path = ".",
    repo: Optional[Neo4jRepository] = None,
    clear_first: bool = True,
) -> Dict[str, Any]:
    annotation_paths = find_annotation_files(annotation_dir)
    logger.info("Found %d annotation files in %s", len(annotation_paths), annotation_dir)

    graph_stats, uri_map = import_graph_json(graph_json_path, repo, clear_first)

    text_stats = {}
    if annotation_paths:
        text_stats = import_text_relations(annotation_paths, repo, uri_map)

    return {**graph_stats, **text_stats,
            "annotation_files": len(annotation_paths)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with Neo4jRepository(
        database_uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="123456789",
    ) as repo:
        stats = import_all(
            graph_json_path="graph.json",
            annotation_dir=".",
            repo=repo,
            clear_first=True,
        )

    print("\n=== Результаты импорта ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
