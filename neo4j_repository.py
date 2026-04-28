
import json
import uuid
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Record
from pydantic import BaseModel, PrivateAttr


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
        self._driver = GraphDatabase.driver(
            self.database_uri, auth=(self.user, self.password)
        )
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute_query(
        self, query: str, params: Dict[str, Any] = None
    ) -> tuple[list[Record], Any]:
        with self._driver.session() as session:
            result = session.run(query, params)
            data = list(result)
            summary = result.consume()
            return data, summary

    def get_all_nodes_and_arcs(
        self,
    ) -> tuple[list[Neo4JData.TNode], list[Neo4JData.TArc]]:
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

    def create_arc(
        self, node1_uri: str, node2_uri: str, rel_type: str = "CONNECTS", **props
    ) -> Neo4JData.TArc:
        props["uri"] = Neo4JData._generate_random_string()
        props_str = Neo4JData._transform_props(props)
        query = f"""
        MATCH (a {{uri: $uri1}}), (b {{uri: $uri2}})
        CREATE (a)-[r:`{rel_type}` {props_str}]->(b)
        RETURN r, a, b
        """
        results, _ = self.execute_query(
            query=query, params={"uri1": node1_uri, "uri2": node2_uri}
        )
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
        _, summary = self.execute_query(
            query=query, params={"element_id": arc_id}
        )
        if not summary:
            raise ValueError("Delete failed")
        return summary.counters.relationships_deleted > 0

    def update_node(
        self, node_uri: str, params: Dict[str, Any]
    ) -> Optional[Neo4JData.TNode]:
        if not params:
            raise ValueError("Params are empty")
        set_params = ", ".join(f"n.`{k}` = $params.`{k}`" for k in params)
        query = f"""
        MATCH (n {{uri: $uri}})
        SET {set_params}
        RETURN n
        """
        results, _ = self.execute_query(
            query=query, params={"uri": node_uri, "params": params}
        )
        if not results:
            raise ValueError("Node not found")
        return Neo4JData._collect_node(results[0])

    def run_custom_query(
        self, query: str, params: Dict[str, Any]
    ) -> Optional[tuple[List[Record], Any]]:
        return self.execute_query(query=query, params=params)
