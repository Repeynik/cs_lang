

from typing import Any, Dict, List, Optional, Tuple

from neo4j import Record
from pydantic import BaseModel

from neo4j_repository import Neo4JData, Neo4jRepository

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

    def _match_nodes_query(
        self, cypher: str, params: Dict[str, Any]
    ) -> List[Neo4JData.TNode]:
        rows, _ = self.repo.execute_query(cypher, params)
        out: List[Neo4JData.TNode] = []
        for r in rows:
            n = r.get("n")
            if n is None:
                continue
            out.append(Neo4JData._collect_node(Record({"n": n})))
        return out

    def _match_arcs_query(
        self, cypher: str, params: Dict[str, Any]
    ) -> List[Neo4JData.TArc]:
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
        return [
            Neo4JData._collect_node(Record({"n": r.get("n")}))
            for r in rows
            if r.get("n") is not None
        ]

    def get_class_children(self, class_uri: str) -> List[Neo4JData.TNode]:
        self._require_label(class_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (ch:`{self.CLASS_LABEL}`)-[:`{self.REL_SUBCLASS_OF}`]->(c:`{self.CLASS_LABEL}` {{uri:$uri}})
        RETURN ch as n
        """
        rows, _ = self.repo.execute_query(q, {"uri": class_uri})
        return [
            Neo4JData._collect_node(Record({"n": r.get("n")}))
            for r in rows
            if r.get("n") is not None
        ]

    def get_class_objects(self, class_uri: str) -> List[Neo4JData.TNode]:
        self._require_label(class_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (o:`{self.OBJECT_LABEL}`)-[:`{self.REL_INSTANCE_OF}`]->(c:`{self.CLASS_LABEL}` {{uri:$uri}})
        RETURN o as n
        """
        rows, _ = self.repo.execute_query(q, {"uri": class_uri})
        return [
            Neo4JData._collect_node(Record({"n": r.get("n")}))
            for r in rows
            if r.get("n") is not None
        ]

    def update_class(
        self,
        class_uri: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        params: Dict[str, Any] = {}
        if name is not None:
            params[self.TITLE_PROP] = name
        if description is not None:
            params["description"] = description
        return self.repo.update_node(class_uri, params)

    def create_class(
        self,
        name: str,
        description: str = "",
        parent_uri: Optional[str] = None,
    ) -> Neo4JData.TNode:
        n = self.repo.create_node(
            {"label": self.CLASS_LABEL, "description": description}
        )
        self.repo.update_node(n.uri, {self.TITLE_PROP: name})
        if parent_uri:
            self.add_class_parent(parent_uri=parent_uri, target_uri=n.uri)
        return self.repo.get_node_by_uri(n.uri)

    def add_class_parent(
        self, parent_uri: str, target_uri: str
    ) -> Neo4JData.TArc:
        self._require_label(parent_uri, self.CLASS_LABEL)
        self._require_label(target_uri, self.CLASS_LABEL)
        q = f"""
        MATCH (t:`{self.CLASS_LABEL}` {{uri:$t}}), (p:`{self.CLASS_LABEL}` {{uri:$p}})
        MERGE (t)-[r:`{self.REL_SUBCLASS_OF}`]->(p)
        ON CREATE SET r.uri = $ruri
        RETURN r
        """
        ruri = Neo4JData._generate_random_string()
        rows, _ = self.repo.execute_query(
            q, {"t": target_uri, "p": parent_uri, "ruri": ruri}
        )
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

    def _collect_props_for_classes(
        self, class_uris: List[str]
    ) -> Tuple[List[str], List[str]]:
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

        deleted_objects = sum(1 for ou in obj_uris if self.repo.delete_node_by_uri(ou))
        deleted_dtps = sum(1 for pu in dtp_uris if self.repo.delete_node_by_uri(pu))
        deleted_ops = sum(1 for pu in op_uris if self.repo.delete_node_by_uri(pu))
        deleted_classes = sum(
            1 for cu in all_classes if self.repo.delete_node_by_uri(cu)
        )

        return {
            "classes_deleted": deleted_classes,
            "objects_deleted": deleted_objects,
            "datatype_properties_deleted": deleted_dtps,
            "object_properties_deleted": deleted_ops,
        }

    def add_class_attribue(
        self,
        class_uri: str,
        attr_name: str,
        datatype: str = "string",
        description: str = "",
    ) -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        prop = self.repo.create_node(
            {"label": self.DATATYPE_PROPERTY_LABEL, "description": description}
        )
        self.repo.update_node(
            prop.uri, {self.ATTR_NAME_PROP: attr_name, self.DATATYPE_PROP: datatype}
        )
        self.repo.create_arc(class_uri, prop.uri, self.REL_HAS_DTP)
        return self.repo.get_node_by_uri(prop.uri)

    def delete_class_attribue(self, datatype_property_uri: str) -> bool:
        self._require_label(datatype_property_uri, self.DATATYPE_PROPERTY_LABEL)
        return self.repo.delete_node_by_uri(datatype_property_uri)

    def add_class_object_attribute(
        self,
        class_uri: str,
        attr_name: str,
        range_class_uri: str,
        description: str = "",
    ) -> Neo4JData.TNode:
        self._require_label(class_uri, self.CLASS_LABEL)
        self._require_label(range_class_uri, self.CLASS_LABEL)
        prop = self.repo.create_node(
            {"label": self.OBJECT_PROPERTY_LABEL, "description": description}
        )
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
            fields.append(
                OntologyModels.SignatureItem(
                    kind="DatatypeProperty",
                    uri=p.get("uri", ""),
                    attr_name=p.get(self.ATTR_NAME_PROP, "") or "",
                    description=p.get("description", "") or "",
                    datatype=p.get(self.DATATYPE_PROP, "") or "",
                )
            )

        for r in op_rows:
            p = r.get("p")
            if p is None:
                continue
            rc = r.get("rc")
            range_uri = ""
            if rc is not None:
                range_uri = rc.get("uri", "") or ""
            fields.append(
                OntologyModels.SignatureItem(
                    kind="ObjectProperty",
                    uri=p.get("uri", ""),
                    attr_name=p.get(self.ATTR_NAME_PROP, "") or "",
                    description=p.get("description", "") or "",
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
        node = self.repo.create_node(
            {"label": self.OBJECT_LABEL, "description": description}
        )
        self.repo.create_arc(node.uri, class_uri, self.REL_INSTANCE_OF)

        datatype_fields = {
            f.attr_name: f for f in sig.fields if f.kind == "DatatypeProperty"
        }
        object_fields = {
            f.attr_name: f for f in sig.fields if f.kind == "ObjectProperty"
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
            if k not in object_fields or v is None:
                continue
            field = object_fields[k]
            target_uris = [v] if isinstance(v, str) else v
            for target_uri in target_uris:
                if not isinstance(target_uri, str):
                    continue
                self._require_label(target_uri, self.OBJECT_LABEL)
                q = f"""
                MATCH (target:`{self.OBJECT_LABEL}` {{uri:$target_uri}})-[:`{self.REL_INSTANCE_OF}`]->(tc:`{self.CLASS_LABEL}`)
                RETURN tc.uri as class_uri
                """
                rows, _ = self.repo.execute_query(q, {"target_uri": target_uri})
                if not rows:
                    raise ValueError(f"Target object `{target_uri}` has no class")
                target_class_uri = rows[0].get("class_uri")
                if (
                    field.range_class_uri
                    and target_class_uri != field.range_class_uri
                ):
                    raise ValueError(
                        f"Object `{target_uri}` is not instance of `{field.range_class_uri}`"
                    )
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
            f.attr_name: f for f in sig.fields if f.kind == "DatatypeProperty"
        }
        object_fields = {
            f.attr_name: f for f in sig.fields if f.kind == "ObjectProperty"
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
            target_uris = [v] if isinstance(v, str) else v
            for target_uri in target_uris:
                if not isinstance(target_uri, str):
                    continue
                self._require_label(target_uri, self.OBJECT_LABEL)
                q2 = f"""
                MATCH (target:`{self.OBJECT_LABEL}` {{uri:$target_uri}})-[:`{self.REL_INSTANCE_OF}`]->(tc:`{self.CLASS_LABEL}`)
                RETURN tc.uri as class_uri
                """
                rows2, _ = self.repo.execute_query(q2, {"target_uri": target_uri})
                if not rows2:
                    raise ValueError(f"Target object `{target_uri}` has no class")
                target_class_uri = rows2[0].get("class_uri")
                if (
                    field.range_class_uri
                    and target_class_uri != field.range_class_uri
                ):
                    raise ValueError(
                        f"Object `{target_uri}` is not instance of `{field.range_class_uri}`"
                    )
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
        chunk_node = self.repo.create_node(
            {"label": self.CHUNK_LABEL, "description": ""}
        )
        self.repo.update_node(
            chunk_node.uri,
            {
                "source_id": source_uri,
                "source_type": source_type,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text": text,
                "embedding": embedding,
                "start_char": start_char,
                "end_char": end_char,
            },
        )
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
            texts.append(
                {"source_id": cls.uri, "source_type": "Class", "text": text}
            )

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
            text = (
                f"Объект: {title}. Описание: {description}. Атрибуты: {payload_text}".strip()
            )
            texts.append(
                {"source_id": obj.uri, "source_type": "Object", "text": text}
            )

        return texts
