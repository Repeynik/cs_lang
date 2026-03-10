

import logging
from cs_lang import Neo4jRepository  


logging.basicConfig(level=logging.INFO)


URI = "neo4j://127.0.0.1:7687"   
USER = "neo4j"
PASSWORD = "123456789"               

def main():
    
    with Neo4jRepository(database_uri=URI, user=USER, password=PASSWORD) as repo:
        
        print("\n--- Создание узлов ---")
        node1 = repo.create_node({"label": "Person", "description": "Алиса", "age": 30})
        node2 = repo.create_node({"label": "Person", "description": "Боб", "age": 25})
        node3 = repo.create_node({"label": "Company", "description": "Технологии будущего", "industry": "IT"})
        print(f"Создан узел: {node1}")
        print(f"Создан узел: {node2}")
        print(f"Создан узел: {node3}")

        
        print("\n--- Создание связей ---")
        arc1 = repo.create_arc(node1.uri, node2.uri, rel_type="KNOWS", since=2020)
        arc2 = repo.create_arc(node1.uri, node3.uri, rel_type="WORKS_FOR", position="Инженер")
        print(f"Создана связь: {arc1}")
        print(f"Создана связь: {arc2}")

        
        print("\n--- Все узлы и связи ---")
        all_nodes, all_arcs = repo.get_all_nodes_and_arcs()
        print(f"Всего узлов: {len(all_nodes)}")
        for n in all_nodes:
            print(f"  {n}")
        print(f"Всего связей: {len(all_arcs)}")
        for r in all_arcs:
            print(f"  {r}")

        
        print("\n--- Узлы с меткой 'Person' ---")
        persons = repo.get_nodes_by_labels(["Person"])
        for p in persons:
            print(f"  {p}")

        
        print("\n--- Получение узла по uri ---")
        try:
            found_node = repo.get_node_by_uri(node1.uri)
            print(f"Найден узел: {found_node}")
        except ValueError:
            print("Узел не найден")

        
        print("\n--- Обновление узла ---")
        updated = repo.update_node(node1.uri, {"age": 31, "city": "Москва"})
        print(f"Обновлённый узел: {updated}")
        
        print("\n--- Удаление ---")
        
        repo.delete_arc_by_id(arc1.element_id)
        repo.delete_arc_by_id(arc2.element_id)
        print("Связи удалены")
        
        repo.delete_node_by_uri(node1.uri)
        repo.delete_node_by_uri(node2.uri)
        repo.delete_node_by_uri(node3.uri)
        print("Узлы удалены")

        
        nodes_after, arcs_after = repo.get_all_nodes_and_arcs()
        print(f"После удаления осталось узлов: {len(nodes_after)}, связей: {len(arcs_after)}")
        print(f"После удаления осталось узлов: {len(nodes_after)}, связей: {len(arcs_after)}")
    with Neo4jRepository(
        database_uri="bolt://localhost:7687",
        user="neo4j",
        password="123456789",
    ) as repo:
        ontology_repo = OntologyRepository(repo=repo)
        embedding_service = EmbeddingService()
        pipeline = OntologyEmbeddingPipeline(
            ontology_repo=ontology_repo,
            embedding_service=embedding_service,
        )

        embedded_chunks = pipeline.save_embeddings_to_neo4j(
            chunk_size=500,
            overlap=100,
        )

        print(f"Saved {len(embedded_chunks)} chunks with embeddings")

        stored_chunks = ontology_repo.get_all_text_chunks()
        search_service = EmbeddingSearchService(embedding_service)

        results = search_service.find_most_similar(
            query="поиск похожих описаний классов",
            stored_chunks=stored_chunks,
            top_k=5,
        )

        for item in results:
            print(item["score"], item["chunk"].get("text", ""))


if __name__ == "__main__":
    main()