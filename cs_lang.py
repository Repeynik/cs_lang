from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from neo4j import GraphDatabase, Record
from pydantic import BaseModel, PrivateAttr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Neo4JData:
    class TNode(BaseModel):
        uri: str
        description: str
        label: str

    class TArc(BaseModel):
        element_id: str
        uri: str
        node_uri_from: str
        node_uri_to: str

    @staticmethod
    def _transform_labels(labels, separator=":") -> str:
        if not labels:
            return ""
        return ":" + separator.join(f"`{l}`" for l in labels)

    @staticmethod
    def _transform_props(props) -> str:
        if not props:
            return "{}"
        items = [f"`{key}`:{json.dumps(value)}" for key, value in props.items()]
        return "{" + ",".join(items) + "}"

    @staticmethod
    def _generate_random_string(namespace: str = "neo4j://127.0.0.1:7687") -> str:
        return f"{namespace}/{uuid.uuid4()}"

    @staticmethod
    def _collect_node(node: Record) -> "Neo4JData.TNode":
        node = node.get("n")
        if not node:
            raise ValueError("there is no node value")
        return Neo4JData.TNode(
            uri=node.get("uri", ""),
            description=node.get("description", ""),
            label=":".join(node.labels),
        )

    @staticmethod
    def _collect_arc(arc) -> "Neo4JData.TArc":
        if isinstance(arc, Record):
            arc = arc.get("r")
            if not arc:
                raise ValueError("there is no arc value in record")
        if not arc:
            raise ValueError("there is no arc value")
        return Neo4JData.TArc(
            element_id=arc.element_id,
            uri=arc.get("uri", ""),
            node_uri_from=arc.start_node.get("uri", ""),
            node_uri_to=arc.end_node.get("uri", ""),
        )


class Neo4jRepository(BaseModel):
    database_uri: str = ""
    user: str = ""
    password: str = ""

    _driver = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._driver = GraphDatabase.driver(self.database_uri, auth=(self.user, self.password))
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> tuple[list[Record], Any]:
        with self._driver.session() as session:
            result = session.run(query, params)
            data = list(result)
            summary = result.consume()
            return data, summary

    def get_all_nodes_and_arcs(self) -> tuple[list[Neo4JData.TNode], list[Neo4JData.TArc]]:
        nodes_query = "MATCH (n) RETURN n"
        nodes_results, _ = self.execute_query(nodes_query)
        nodes = [Neo4JData._collect_node(record) for record in nodes_results]

        arcs_query = "MATCH ()-[r]-() RETURN r"
        arcs_results, _ = self.execute_query(arcs_query)
        arcs = [Neo4JData._collect_arc(record) for record in arcs_results]

        return nodes, arcs

    def get_nodes_by_labels(self, labels: list[str]) -> list[Neo4JData.TNode]:
        query = f"MATCH (n{Neo4JData._transform_labels(labels=labels)}) RETURN n"
        results, _ = self.execute_query(query=query)
        return [Neo4JData._collect_node(node=node) for node in results]

    def get_node_by_uri(self, node_uri: str) -> Optional[Neo4JData.TNode]:
        query = "MATCH (n {uri : $uri }) RETURN n"
        results, _ = self.execute_query(query=query, params={"uri": node_uri})
        if not results:
            raise ValueError("Node not found")
        return Neo4JData._collect_node(node=results[0])

    def create_node(self, params: dict[str, Any]) -> Neo4JData.TNode:
        label = params.get("label", "Node")
        description = params.get("description", "")
        uri = Neo4JData._generate_random_string()
        props = {"uri": uri, "description": description}
        props_str = Neo4JData._transform_props(props)
        query = f"CREATE (n:`{label}` {props_str}) RETURN n"
        results, _ = self.execute_query(query=query)
        if not results:
            raise ValueError("Node not created")
        return Neo4JData._collect_node(results[0])

    def create_arc(self, node1_uri: str, node2_uri: str, rel_type: str = "CONNECTS", **props) -> Neo4JData.TArc:
        props["uri"] = Neo4JData._generate_random_string()
        props_str = Neo4JData._transform_props(props)
        query = f"""
        MATCH (a {{uri: $uri1}}), (b {{uri: $uri2}})
        CREATE (a)-[r:`{rel_type}` {props_str}]->(b)
        RETURN r, a, b
        """
        results, _ = self.execute_query(query=query, params={"uri1": node1_uri, "uri2": node2_uri})
        if not results:
            raise ValueError("Arc not created")
        return Neo4JData._collect_arc(results[0])

    def delete_node_by_uri(self, node_uri: str) -> bool:
        query = "MATCH (n {uri : $uri }) DETACH DELETE n"
        _, summary = self.execute_query(query=query, params={"uri": node_uri})
        if not summary:
            raise ValueError("Delete failed")
        return summary.counters.nodes_deleted > 0

    def delete_arc_by_id(self, arc_id: str) -> bool:
        query = "MATCH ()-[r]->() WHERE r.elementId = $element_id DELETE r"
        _, summary = self.execute_query(query=query, params={"element_id": arc_id})
        if not summary:
            raise ValueError("Delete failed")
        return summary.counters.relationships_deleted > 0

    def update_node(self, node_uri: str, params: Dict[str, Any]) -> Optional[Neo4JData.TNode]:
        if not params:
            raise ValueError("Params are empty")
        set_params = ", ".join(f"n.`{k}` = $params.`{k}`" for k in params)
        query = f"""
        MATCH (n {{uri: $uri}})
        SET {set_params}
        RETURN n
        """
        results, _ = self.execute_query(query=query, params={"uri": node_uri, "params": params})
        if not results:
            raise ValueError("Node not found")
        return Neo4JData._collect_node(results[0])

    def run_custom_query(self, query: str, params: Dict[str, Any]) -> Optional[tuple[List[Record], Any]]:
        return self.execute_query(query=query, params=params)


class OntologyModels:
    class ClassNode(BaseModel):
        uri: str
        title: str
        description: str

    class ObjectNode(BaseModel):
        uri: str
        title: str
        description: str
        class_uri: str

    class DatatypePropertyNode(BaseModel):
        uri: str
        attr_name: str
        description: str
        datatype: str

    class ObjectPropertyNode(BaseModel):
        uri: str
        attr_name: str
        description: str
        range_class_uri: str

    class SignatureItem(BaseModel):
        kind: str
        uri: str
        attr_name: str
        description: str = ""
        datatype: Optional[str] = None
        range_class_uri: Optional[str] = None

    class Signature(BaseModel):
        class_uri: str
        title: str
        description: str
        fields: List["OntologyModels.SignatureItem"]


class ChunkModels:
    class TextChunk(BaseModel):
        chunk_id: str
        source_id: str
        source_type: str
        text: str
        chunk_index: int
        start_char: int
        end_char: int


class OntologyRepository(BaseModel):
    repo: Neo4jRepository

    CLASS_LABEL: str = "Class"
    OBJECT_LABEL: str = "Object"
    DATATYPE_PROPERTY_LABEL: str = "DatatypeProperty"
    OBJECT_PROPERTY_LABEL: str = "ObjectProperty"
    CHUNK_LABEL: str = "TextChunk"

    REL_SUBCLASS_OF: str = "SUBCLASS_OF"
    REL_HAS_DTP: str = "HAS_DATATYPE_PROPERTY"
    REL_HAS_OP: str = "HAS_OBJECT_PROPERTY"
    REL_INSTANCE_OF: str = "INSTANCE_OF"
    REL_OP_RANGE: str = "RANGE"
    REL_HAS_CHUNK: str = "HAS_CHUNK"

    TITLE_PROP: str = "title"
    ATTR_NAME_PROP: str = "attr_name"
    DATATYPE_PROP: str = "datatype"

    def _collect_class(self, node: Neo4JData.TNode) -> OntologyModels.ClassNode:
        props = self._node_props(node.uri)
        return OntologyModels.ClassNode(
            uri=node.uri,
            title=str(props.get(self.TITLE_PROP, "")),
            description=str(props.get("description", "")),
        )

    def _node_props(self, node_uri: str) -> Dict[str, Any]:
        q = "MATCH (n {uri:$uri}) RETURN properties(n) as p, labels(n) as l"
        res, _ = self.repo.execute_query(q, {"uri": node_uri})
        if not res:
            raise ValueError("Node not found")
        p = res[0].get("p") or {}
        l = res[0].get("l") or []
        p["_labels"] = l
        return p

    def _require_label(self, node_uri: str, label: str) -> None:
        props = self._node_props(node_uri)
        labels = props.get("_labels", [])
        if label not in labels:
            raise ValueError(f"Node {node_uri} is not labeled `{label}`")

    def _match_nodes_query(self, cypher: str, params: Dict[str, Any]) -> List[Neo4JData.TNode]:
        rows, _ = self.repo.execute_query(cypher, params)
        out: List[Neo4JData.TNode] = []
        for r in rows:
            n = r.get("n")
            if n is None:
                continue
            out.append(Neo4JData._collect_node(Record({"n": n})))
        return out

    def _match_arcs_query(self, cypher: str, params: Dict[str, Any]) -> List[Neo4JData.TArc]:
        rows, _ = self.repo.execute_query(cypher, params)
        out: List[Neo4JData.TArc] = []
        for r in rows:
            rel = r.get("r")
            if rel is None:
                continue
            out.append(Neo4JData._collect_arc(Record({"r": rel})))
        return out

    def get_ontology(self) -> Dict[str, List[Any]]:
        classes_q = f"MATCH (n:`{self.CLASS_LABEL}`) RETURN n"
        objects_q = f"MATCH (n:`{self.OBJECT_LABEL}`) RETURN n"
        dtp_q = f"MATCH (n:`{self.DATATYPE_PROPERTY_LABEL}`) RETURN n"
        op_q = f"MATCH (n:`{self.OBJECT_PROPERTY_LABEL}`) RETURN n"
        rels_q = "MATCH ()-[r]-() RETURN r"

        classes = self._match_nodes_query(classes_q, {})
        objects = self._match_nodes_query(objects_q, {})
        dtps = self._match_nodes_query(dtp_q, {})
        ops = self._match_nodes_query(op_q, {})
        rels = self._match_arcs_query(rels_q, {})

        return {
            "classes": classes,
            "objects": objects,
            "datatype_properties": dtps,
            "object_properties": ops,
            "relations": rels,
        }

    def get_ontology_parent_classes(self) -> List[Neo4JData.TNode]:
        q = f"""
        MATCH (n:`{self.CLASS_LABEL}`)
        WHERE NOT (n)-[:`{self.REL_SUBCLASS_OF}`]->(:`{self.CLASS_LABEL}`)
        RETURN n
        """
        return self._match_nodes_query(q, {})

    def get_class(self, class_uri: str) -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        return self.repo.get_node_by_uri(class_uri)

    def get_class_parents(self, class_uri: str) -> List[Neo4JData.TNode]:
        self._require_label(class_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (c:`{self.CLASS_LABEL}` {{uri:$uri}})-[:`{self.REL_SUBCLASS_OF}`]->(p:`{self.CLASS_LABEL}`)
        RETURN p as n
        """
        rows, _ = self.repo.execute_query(q, {"uri": class_uri})
        out: List[Neo4JData.TNode] = []
        for r in rows:
            n = r.get("n")
            if n is None:
                continue
            out.append(Neo4JData._collect_node(Record({"n": n})))
        return out

    def get_class_children(self, class_uri: str) -> List[Neo4JData.TNode]:
        self._require_label(class_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (ch:`{self.CLASS_LABEL}`)-[:`{self.REL_SUBCLASS_OF}`]->(c:`{self.CLASS_LABEL}` {{uri:$uri}})
        RETURN ch as n
        """
        rows, _ = self.repo.execute_query(q, {"uri": class_uri})
        out: List[Neo4JData.TNode] = []
        for r in rows:
            n = r.get("n")
            if n is None:
                continue
            out.append(Neo4JData._collect_node(Record({"n": n})))
        return out

    def get_class_objects(self, class_uri: str) -> List[Neo4JData.TNode]:
        self._require_label(class_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (o:`{self.OBJECT_LABEL}`)-[:`{self.REL_INSTANCE_OF}`]->(c:`{self.CLASS_LABEL}` {{uri:$uri}})
        RETURN o as n
        """
        rows, _ = self.repo.execute_query(q, {"uri": class_uri})
        out: List[Neo4JData.TNode] = []
        for r in rows:
            n = r.get("n")
            if n is None:
                continue
            out.append(Neo4JData._collect_node(Record({"n": n})))
        return out

    def update_class(self, class_uri: str, name: Optional[str] = None, description: Optional[str] = None) -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        params: Dict[str, Any] = {}
        if name is not None:
            params[self.TITLE_PROP] = name
        if description is not None:
            params["description"] = description
        return self.repo.update_node(class_uri, params)

    def create_class(self, name: str, description: str = "", parent_uri: Optional[str] = None) -> Neo4JData.TNode:
        n = self.repo.create_node({"label": self.CLASS_LABEL, "description": description})
        self.repo.update_node(n.uri, {self.TITLE_PROP: name})
        if parent_uri:
            self.add_class_parent(parent_uri=parent_uri, target_uri=n.uri)
        return self.repo.get_node_by_uri(n.uri)

    def add_class_parent(self, parent_uri: str, target_uri: str) -> Neo4JData.TArc:
        self._require_label(parent_uri, self.CLASS_LABEL)
        self._require_label(target_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (t:`{self.CLASS_LABEL}` {{uri:$t}}), (p:`{self.CLASS_LABEL}` {{uri:$p}})
        MERGE (t)-[r:`{self.REL_SUBCLASS_OF}`]->(p)
        ON CREATE SET r.uri = $ruri
        RETURN r
        """
        ruri = Neo4JData._generate_random_string()
        rows, _ = self.repo.execute_query(q, {"t": target_uri, "p": parent_uri, "ruri": ruri})
        if not rows:
            raise ValueError("Parent link not created")
        return Neo4JData._collect_arc(Record({"r": rows[0].get("r")}))

    def _collect_descendants(self, class_uri: str) -> List[str]:
        q = f"""
        MATCH (c:`{self.CLASS_LABEL}` {{uri:$uri}})
        MATCH (d:`{self.CLASS_LABEL}`)-[:`{self.REL_SUBCLASS_OF}`*1..]->(c)
        RETURN DISTINCT d.uri as uri
        """
        rows, _ = self.repo.execute_query(q, {"uri": class_uri})
        return [r.get("uri") for r in rows if r.get("uri")]

    def _collect_objects_for_classes(self, class_uris: List[str]) -> List[str]:
        if not class_uris:
            return []
        q = f"""
        MATCH (o:`{self.OBJECT_LABEL}`)-[:`{self.REL_INSTANCE_OF}`]->(c:`{self.CLASS_LABEL}`)
        WHERE c.uri IN $uris
        RETURN DISTINCT o.uri as uri
        """
        rows, _ = self.repo.execute_query(q, {"uris": class_uris})
        return [r.get("uri") for r in rows if r.get("uri")]

    def _collect_props_for_classes(self, class_uris: List[str]) -> Tuple[List[str], List[str]]:
        if not class_uris:
            return [], []
        dtp_q = f"""
        MATCH (c:`{self.CLASS_LABEL}`)-[:`{self.REL_HAS_DTP}`]->(p:`{self.DATATYPE_PROPERTY_LABEL}`)
        WHERE c.uri IN $uris
        RETURN DISTINCT p.uri as uri
        """
        op_q = f"""
        MATCH (c:`{self.CLASS_LABEL}`)-[:`{self.REL_HAS_OP}`]->(p:`{self.OBJECT_PROPERTY_LABEL}`)
        WHERE c.uri IN $uris
        RETURN DISTINCT p.uri as uri
        """
        dtp_rows, _ = self.repo.execute_query(dtp_q, {"uris": class_uris})
        op_rows, _ = self.repo.execute_query(op_q, {"uris": class_uris})
        dtp_uris = [r.get("uri") for r in dtp_rows if r.get("uri")]
        op_uris = [r.get("uri") for r in op_rows if r.get("uri")]
        return dtp_uris, op_uris

    def delete_class(self, class_uri: str) -> Dict[str, int]:
        self._require_label(class_uri, self.CLASS_LABEL)
        descendants = self._collect_descendants(class_uri)
        all_classes = [class_uri] + [u for u in descendants if u != class_uri]

        obj_uris = self._collect_objects_for_classes(all_classes)
        dtp_uris, op_uris = self._collect_props_for_classes(all_classes)

        deleted_objects = 0
        for ou in obj_uris:
            if self.repo.delete_node_by_uri(ou):
                deleted_objects += 1

        deleted_dtps = 0
        for pu in dtp_uris:
            if self.repo.delete_node_by_uri(pu):
                deleted_dtps += 1

        deleted_ops = 0
        for pu in op_uris:
            if self.repo.delete_node_by_uri(pu):
                deleted_ops += 1

        deleted_classes = 0
        for cu in all_classes:
            if self.repo.delete_node_by_uri(cu):
                deleted_classes += 1

        return {
            "classes_deleted": deleted_classes,
            "objects_deleted": deleted_objects,
            "datatype_properties_deleted": deleted_dtps,
            "object_properties_deleted": deleted_ops,
        }

    def add_class_attribue(self, class_uri: str, attr_name: str, datatype: str = "string", description: str = "") -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        prop = self.repo.create_node({"label": self.DATATYPE_PROPERTY_LABEL, "description": description})
        self.repo.update_node(prop.uri, {self.ATTR_NAME_PROP: attr_name, self.DATATYPE_PROP: datatype})
        self.repo.create_arc(class_uri, prop.uri, self.REL_HAS_DTP)
        return self.repo.get_node_by_uri(prop.uri)

    def delete_class_attribue(self, datatype_property_uri: str) -> bool:
        self._require_label(datatype_property_uri, self.DATATYPE_PROPERTY_LABEL)
        return self.repo.delete_node_by_uri(datatype_property_uri)

    def add_class_object_attribute(self, class_uri: str, attr_name: str, range_class_uri: str, description: str = "") -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        self._require_label(range_class_uri, self.CLASS_LABEL)
        prop = self.repo.create_node({"label": self.OBJECT_PROPERTY_LABEL, "description": description})
        self.repo.update_node(prop.uri, {self.ATTR_NAME_PROP: attr_name})
        self.repo.create_arc(class_uri, prop.uri, self.REL_HAS_OP)
        self.repo.create_arc(prop.uri, range_class_uri, self.REL_OP_RANGE)
        return self.repo.get_node_by_uri(prop.uri)

    def delete_class_object_attribute(self, object_property_uri: str) -> bool:
        self._require_label(object_property_uri, self.OBJECT_PROPERTY_LABEL)
        return self.repo.delete_node_by_uri(object_property_uri)

    def get_object(self, object_uri: str) -> Dict[str, Any]:
        self._require_label(object_uri, self.OBJECT_LABEL)
        node = self.repo.get_node_by_uri(object_uri)

        class_q = f"""
        MATCH (o:`{self.OBJECT_LABEL}` {{uri:$uri}})-[:`{self.REL_INSTANCE_OF}`]->(c:`{self.CLASS_LABEL}`)
        RETURN c.uri as class_uri
        """
        rows, _ = self.repo.execute_query(class_q, {"uri": object_uri})
        if not rows:
            raise ValueError("Object has no class")

        class_uri = rows[0].get("class_uri")
        sig = self.collect_signature(class_uri)
        props = self._node_props(object_uri)

        result: Dict[str, Any] = {
            "uri": node.uri,
            "description": node.description,
            "class_uri": class_uri,
            "data": {},
        }

        for field in sig.fields:
            if field.kind == "DatatypeProperty":
                result["data"][field.attr_name] = props.get(field.attr_name)
            elif field.kind == "ObjectProperty":
                q = f"""
                MATCH (o:`{self.OBJECT_LABEL}` {{uri:$uri}})-[:`{field.attr_name}`]->(target:`{self.OBJECT_LABEL}`)
                RETURN target.uri as uri
                """
                rel_rows, _ = self.repo.execute_query(q, {"uri": object_uri})
                uris = [r.get("uri") for r in rel_rows if r.get("uri")]
                result["data"][field.attr_name] = uris

        return result

    def delete_object(self, object_uri: str) -> bool:
        self._require_label(object_uri, self.OBJECT_LABEL)
        return self.repo.delete_node_by_uri(object_uri)

    def collect_signature(self, class_uri: str) -> OntologyModels.Signature:
        self._require_label(class_uri, self.CLASS_LABEL)

        class_props = self._node_props(class_uri)
        title = class_props.get(self.TITLE_PROP, "")
        description = class_props.get("description", "")

        dtp_q = f"""
        MATCH (c:`{self.CLASS_LABEL}` {{uri:$uri}})-[:`{self.REL_HAS_DTP}`]->(p:`{self.DATATYPE_PROPERTY_LABEL}`)
        RETURN p
        """
        op_q = f"""
        MATCH (c:`{self.CLASS_LABEL}` {{uri:$uri}})-[:`{self.REL_HAS_OP}`]->(p:`{self.OBJECT_PROPERTY_LABEL}`)
        OPTIONAL MATCH (p)-[:`{self.REL_OP_RANGE}`]->(rc:`{self.CLASS_LABEL}`)
        RETURN p, rc
        """

        dtp_rows, _ = self.repo.execute_query(dtp_q, {"uri": class_uri})
        op_rows, _ = self.repo.execute_query(op_q, {"uri": class_uri})

        fields: List[OntologyModels.SignatureItem] = []

        for r in dtp_rows:
            p = r.get("p")
            if p is None:
                continue
            p_uri = p.get("uri", "")
            p_desc = p.get("description", "") or ""
            attr_name = p.get(self.ATTR_NAME_PROP, "") or ""
            datatype = p.get(self.DATATYPE_PROP, "") or ""
            fields.append(
                OntologyModels.SignatureItem(
                    kind="DatatypeProperty",
                    uri=p_uri,
                    attr_name=attr_name,
                    description=p_desc,
                    datatype=datatype,
                )
            )

        for r in op_rows:
            p = r.get("p")
            if p is None:
                continue
            rc = r.get("rc")
            p_uri = p.get("uri", "")
            p_desc = p.get("description", "") or ""
            attr_name = p.get(self.ATTR_NAME_PROP, "") or ""
            range_uri = ""
            if rc is not None:
                range_uri = rc.get("uri", "") or ""
            fields.append(
                OntologyModels.SignatureItem(
                    kind="ObjectProperty",
                    uri=p_uri,
                    attr_name=attr_name,
                    description=p_desc,
                    range_class_uri=range_uri or None,
                )
            )

        return OntologyModels.Signature(
            class_uri=class_uri,
            title=title,
            description=description,
            fields=fields,
        )

    def create_object(
        self,
        class_uri: str,
        obj_params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Neo4JData.TNode:
        sig = self.collect_signature(class_uri)

        obj_params = obj_params or {}
        data = data or {}

        description = str(obj_params.get("description", ""))
        node = self.repo.create_node({
            "label": self.OBJECT_LABEL,
            "description": description,
        })

        self.repo.create_arc(node.uri, class_uri, self.REL_INSTANCE_OF)

        datatype_fields = {
            field.attr_name: field
            for field in sig.fields
            if field.kind == "DatatypeProperty"
        }

        object_fields = {
            field.attr_name: field
            for field in sig.fields
            if field.kind == "ObjectProperty"
        }

        scalar_params: Dict[str, Any] = {}

        for k, v in obj_params.items():
            if k in {"label", "uri", "description"}:
                continue
            scalar_params[k] = v

        for k, v in data.items():
            if k in datatype_fields:
                scalar_params[k] = v

        if scalar_params:
            self.repo.update_node(node.uri, scalar_params)

        for k, v in data.items():
            if k not in object_fields:
                continue

            field = object_fields[k]

            if v is None:
                continue

            if isinstance(v, list):
                target_uris = [item for item in v if isinstance(item, str)]
            elif isinstance(v, str):
                target_uris = [v]
            else:
                raise ValueError(f"ObjectProperty `{k}` must be string or list of strings")

            for target_uri in target_uris:
                self._require_label(target_uri, self.OBJECT_LABEL)

                q = f"""
                MATCH (target:`{self.OBJECT_LABEL}` {{uri:$target_uri}})-[:`{self.REL_INSTANCE_OF}`]->(target_class:`{self.CLASS_LABEL}`)
                RETURN target_class.uri as class_uri
                """
                rows, _ = self.repo.execute_query(q, {"target_uri": target_uri})
                if not rows:
                    raise ValueError(f"Target object `{target_uri}` has no class")

                target_class_uri = rows[0].get("class_uri")
                if field.range_class_uri and target_class_uri != field.range_class_uri:
                    raise ValueError(f"Object `{target_uri}` is not instance of required class `{field.range_class_uri}`")

                self.repo.create_arc(node.uri, target_uri, k)

        return self.repo.get_node_by_uri(node.uri)

    def update_object(
        self,
        object_uri: str,
        obj_params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Neo4JData.TNode:
        self._require_label(object_uri, self.OBJECT_LABEL)

        q = f"""
        MATCH (o:`{self.OBJECT_LABEL}` {{uri:$uri}})-[:`{self.REL_INSTANCE_OF}`]->(c:`{self.CLASS_LABEL}`)
        RETURN c.uri as class_uri
        """
        rows, _ = self.repo.execute_query(q, {"uri": object_uri})
        if not rows:
            raise ValueError("Object has no class")

        class_uri = rows[0].get("class_uri")
        if not class_uri:
            raise ValueError("Object has no class")

        sig = self.collect_signature(class_uri)

        obj_params = obj_params or {}
        data = data or {}

        datatype_fields = {
            field.attr_name: field
            for field in sig.fields
            if field.kind == "DatatypeProperty"
        }

        object_fields = {
            field.attr_name: field
            for field in sig.fields
            if field.kind == "ObjectProperty"
        }

        scalar_params: Dict[str, Any] = {}

        for k, v in obj_params.items():
            if k in {"label", "uri"}:
                continue
            scalar_params[k] = v

        for k, v in data.items():
            if k in datatype_fields:
                scalar_params[k] = v

        if scalar_params:
            self.repo.update_node(object_uri, scalar_params)

        for k, v in data.items():
            if k not in object_fields:
                continue

            delete_q = f"""
            MATCH (o:`{self.OBJECT_LABEL}` {{uri:$object_uri}})-[r:`{k}`]->(:`{self.OBJECT_LABEL}`)
            DELETE r
            """
            self.repo.execute_query(delete_q, {"object_uri": object_uri})

            if v is None:
                continue

            field = object_fields[k]

            if isinstance(v, list):
                target_uris = [item for item in v if isinstance(item, str)]
            elif isinstance(v, str):
                target_uris = [v]
            else:
                raise ValueError(f"ObjectProperty `{k}` must be string or list of strings")

            for target_uri in target_uris:
                self._require_label(target_uri, self.OBJECT_LABEL)

                q = f"""
                MATCH (target:`{self.OBJECT_LABEL}` {{uri:$target_uri}})-[:`{self.REL_INSTANCE_OF}`]->(target_class:`{self.CLASS_LABEL}`)
                RETURN target_class.uri as class_uri
                """
                rows, _ = self.repo.execute_query(q, {"target_uri": target_uri})
                if not rows:
                    raise ValueError(f"Target object `{target_uri}` has no class")

                target_class_uri = rows[0].get("class_uri")
                if field.range_class_uri and target_class_uri != field.range_class_uri:
                    raise ValueError(f"Object `{target_uri}` is not instance of required class `{field.range_class_uri}`")

                self.repo.create_arc(object_uri, target_uri, k)

        return self.repo.get_node_by_uri(object_uri)

    def save_text_chunk(
        self,
        source_uri: str,
        source_type: str,
        chunk_id: str,
        chunk_index: int,
        text: str,
        embedding: List[float],
        start_char: int,
        end_char: int,
    ) -> Neo4JData.TNode:
        chunk_node = self.repo.create_node({
            "label": self.CHUNK_LABEL,
            "description": "",
        })

        self.repo.update_node(chunk_node.uri, {
            "source_id": source_uri,
            "source_type": source_type,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "text": text,
            "embedding": embedding,
            "start_char": start_char,
            "end_char": end_char,
        })

        self.repo.create_arc(source_uri, chunk_node.uri, self.REL_HAS_CHUNK)
        return self.repo.get_node_by_uri(chunk_node.uri)

    def get_all_text_chunks(self) -> List[Dict[str, Any]]:
        q = f"""
        MATCH (c:`{self.CHUNK_LABEL}`)
        RETURN properties(c) as p
        ORDER BY p.source_type, p.source_id, p.chunk_index
        """
        rows, _ = self.repo.execute_query(q, {})
        return [r.get("p") for r in rows if r.get("p")]

    def collect_ontology_texts(self) -> List[Dict[str, Any]]:
        ontology = self.get_ontology()
        texts: List[Dict[str, Any]] = []

        for cls in ontology.get("classes", []):
            class_props = self._node_props(cls.uri)
            title = str(class_props.get("title", ""))
            description = str(class_props.get("description", ""))
            text = f"Класс: {title}. Описание: {description}".strip()
            texts.append({
                "source_id": cls.uri,
                "source_type": "Class",
                "text": text,
            })

        for obj in ontology.get("objects", []):
            obj_props = self._node_props(obj.uri)
            description = str(obj_props.get("description", ""))
            title = str(obj_props.get("title", ""))
            payload_parts = []
            for k, v in obj_props.items():
                if k in {"uri", "description", "_labels"}:
                    continue
                payload_parts.append(f"{k}: {v}")
            payload_text = "; ".join(payload_parts)
            text = f"Объект: {title}. Описание: {description}. Атрибуты: {payload_text}".strip()
            texts.append({
                "source_id": obj.uri,
                "source_type": "Object",
                "text": text,
            })

        return texts


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def get_chunks(
        self,
        texts: List[Dict[str, Any]],
        chunk_size: int = 500,
        overlap: int = 100,
        min_chunk_length: int = 30,
    ) -> List[ChunkModels.TextChunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        result: List[ChunkModels.TextChunk] = []

        for item in texts:
            source_id = str(item.get("source_id", "unknown"))
            source_type = str(item.get("source_type", "unknown"))
            text = str(item.get("text", "")).strip()

            if not text:
                continue

            normalized_text = self._normalize_text(text)
            chunk_spans = self._split_text_into_spans(
                normalized_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for idx, (start, end) in enumerate(chunk_spans):
                chunk_text = normalized_text[start:end].strip()
                if len(chunk_text) < min_chunk_length:
                    continue

                result.append(
                    ChunkModels.TextChunk(
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

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def cos_compare(self, embedding1: List[float], embedding2: List[float]) -> float:
        if not embedding1 or not embedding2:
            raise ValueError("Embeddings must not be empty")

        v1 = np.array(embedding1).reshape(1, -1)
        v2 = np.array(embedding2).reshape(1, -1)

        if v1.shape[1] != v2.shape[1]:
            raise ValueError("Embeddings must have the same dimension")

        score = cosine_similarity(v1, v2)[0][0]
        return float(score)

    def embed_chunks(self, chunks: List[ChunkModels.TextChunk]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)

        result: List[Dict[str, Any]] = []
        for chunk, emb in zip(chunks, embeddings):
            result.append({
                "chunk": chunk,
                "embedding": emb,
            })
        return result

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text_into_spans(self, text: str, chunk_size: int, overlap: int) -> List[tuple[int, int]]:
        spans: List[tuple[int, int]] = []
        text_length = len(text)

        start = 0
        while start < text_length:
            raw_end = min(start + chunk_size, text_length)

            if raw_end < text_length:
                better_end = self._find_good_split_position(text, start, raw_end)
                end = better_end if better_end > start else raw_end
            else:
                end = raw_end

            spans.append((start, end))

            if end >= text_length:
                break

            start = max(0, end - overlap)

        return spans

    def _find_good_split_position(self, text: str, start: int, raw_end: int) -> int:
        window_start = max(start, raw_end - 120)
        candidate = text[window_start:raw_end]

        for pattern in ["\n\n", ". ", "! ", "? ", "; ", ", ", " "]:
            pos = candidate.rfind(pattern)
            if pos != -1:
                return window_start + pos + len(pattern)

        return raw_end


class OntologyEmbeddingPipeline:
    def __init__(self, ontology_repo: OntologyRepository, embedding_service: EmbeddingService):
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
        return self.embedding_service.embed_chunks(chunks)

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


class EmbeddingSearchService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def find_most_similar(
        self,
        query: str,
        stored_chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        query_embedding = self.embedding_service.get_embeddings([query])[0]

        scored: List[Dict[str, Any]] = []
        for chunk_data in stored_chunks:
            emb = chunk_data.get("embedding")
            if not emb:
                continue

            score = self.embedding_service.cos_compare(query_embedding, emb)
            scored.append({
                "score": score,
                "chunk": chunk_data,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
