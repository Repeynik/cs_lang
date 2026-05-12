import logging
from cs_lang import (
    EmbeddingService, EmbeddingSearchService,
    OntologyRepository, Neo4jRepository, OntologyEmbeddingPipeline,
)
from huggingface_hub import snapshot_download
from json_to_neo4j import import_all
from json_ontology_loader import load_ontology_from_json, build_ontology_texts
from text_json_loader import find_annotation_files, load_text_annotations, build_texts_from_annotations
from rag_pipeline import RAGPipeline
from search_service import SearchService
from llm_service import StubLLM

logging.basicConfig(level=logging.INFO)

NEO4J_URI       = "bolt://127.0.0.1:7687"
NEO4J_USER      = "neo4j"
NEO4J_PASS      = "123456789"
EMBEDDING_MODEL = "./paraphrase-multilingual-mpnet-base-v2"
GRAPH_JSON      = "graph.json"
ANNOTATION_DIR  = "."


def example_cosine_similarity():
    service = EmbeddingService(model_name=EMBEDDING_MODEL)
    t1 = "Кошка сидит на подоконнике"
    t2 = "На окне сидит кот"
    t3 = "Сервер не отвечает по SSH"
    embs = service.get_embeddings([t1, t2, t3])
    s12 = service.cos_compare(embs[0], embs[1])
    s13 = service.cos_compare(embs[0], embs[2])
    s23 = service.cos_compare(embs[1], embs[2])
    print("Косинусное сходство:")
    print(f"  text1 vs text2 = {s12:.4f}")
    print(f"  text1 vs text3 = {s13:.4f}")
    print(f"  text2 vs text3 = {s23:.4f}")
    print(f"  Ближе: {'text1 и text2' if s12 > s13 else 'text1 и text3'}")


def example_chunk_embedding_and_search():
    service = EmbeddingService(model_name=EMBEDDING_MODEL)
    search  = EmbeddingSearchService(service)
    texts = [
        {"source_id": "c1", "source_type": "Class",
         "text": "Класс: Сервер. Описание: предоставляет вычислительные ресурсы."},
        {"source_id": "c2", "source_type": "Class",
         "text": "Класс: База данных. Описание: хранит структурированную информацию."},
        {"source_id": "o1", "source_type": "Object",
         "text": "Объект: web-01. Веб сервер production. ip: 10.0.0.1; role: frontend"},
    ]
    chunks   = service.get_chunks(texts=texts, chunk_size=120, overlap=20, min_chunk_length=20)
    embedded = service.embed_chunks(chunks)
    query    = "сервер для веб приложения"
    results  = search.find_most_similar(query=query, stored_chunks=embedded, top_k=3)
    print(f"Поиск: '{query}'")
    for i, r in enumerate(results, 1):
        ch = r["chunk"]["chunk"]
        print(f"  {i}. score={r['score']:.4f} | {ch.text[:70]}...")


def seed_demo_ontology():
    with Neo4jRepository(database_uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS) as repo:
        ontology_repo = OntologyRepository(repo=repo)
        server_class = ontology_repo.create_class(name="Сервер", description="Вычислительный узел")
        db_class     = ontology_repo.create_class(name="База данных", description="Хранение данных")
        ontology_repo.add_class_attribue(class_uri=server_class.uri, attr_name="ip", datatype="string")
        ontology_repo.add_class_attribue(class_uri=server_class.uri, attr_name="role", datatype="string")
        ontology_repo.add_class_object_attribute(
            class_uri=server_class.uri, attr_name="uses_database", range_class_uri=db_class.uri)
        web_obj = ontology_repo.create_object(
            class_uri=server_class.uri,
            obj_params={"description": "Production web server", "title": "web-01"},
            data={"ip": "10.0.0.1", "role": "frontend"})
        db_obj = ontology_repo.create_object(
            class_uri=db_class.uri,
            obj_params={"description": "Primary database", "title": "db-01"},
            data={})
        ontology_repo.update_object(object_uri=web_obj.uri, data={"uses_database": db_obj.uri})
        print("Тестовая онтология создана")


def example_pipeline_with_neo4j():
    with Neo4jRepository(database_uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS) as repo:
        ontology_repo     = OntologyRepository(repo=repo)
        embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
        pipeline = OntologyEmbeddingPipeline(
            ontology_repo=ontology_repo, embedding_service=embedding_service)
        embedded_chunks = pipeline.build_embeddings_for_ontology(chunk_size=300, overlap=50)
        print("Количество подготовленных чанков:", len(embedded_chunks))
        if len(embedded_chunks) >= 2:
            score = embedding_service.cos_compare(
                embedded_chunks[0]["embedding"], embedded_chunks[1]["embedding"])
            print("Cosine similarity между первыми двумя чанками:", round(score, 4))
        pipeline.save_embeddings_to_neo4j(chunk_size=300, overlap=50)
        stored_chunks = ontology_repo.get_all_text_chunks()
        query = "класс для хранения данных"
        query_emb = embedding_service.get_embeddings([query])[0]
        scored = []
        for chunk_data in stored_chunks:
            emb = chunk_data.get("embedding")
            if not emb: continue
            score = embedding_service.cos_compare(query_emb, emb)
            scored.append({"score": score, "text": chunk_data.get("text",""),
                           "source_id": chunk_data.get("source_id",""),
                           "source_type": chunk_data.get("source_type","")})
        scored.sort(key=lambda x: x["score"], reverse=True)
        print("Результаты поиска по embedding в Neo4j:")
        for item in scored[:5]:
            print(f"  score={item['score']:.4f} | {item['source_type']} | {item['text'][:80]}")


def _print_rag(result: dict) -> None:
    print(f"\nВопрос: {result['question']}")
    print("-- Фаза 2 --")
    print(result["intermediate_answer"])
    print("-- Фаза 3 (финал) --")
    print(result["final_answer"])
    for c in result["phase3_chunks"][:4]:
        print(f"  [{c['score']:.4f}] {c['text'][:90]}...")


def example_import_and_rag():

    embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)

    with Neo4jRepository(database_uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASS) as repo:

        print("=" * 60)
        print("ШАГ 1: Импорт graph.json + annotation JSON → Neo4j")
        stats = import_all(GRAPH_JSON, ANNOTATION_DIR, repo, clear_first=True)
        print(f"  Классов: {stats.get('classes',0)}, "
              f"Экземпляров: {stats.get('individuals',0)}, "
              f"Отношений из текста: {stats.get('text_relations',0)}, "
              f"Файлов аннотаций: {stats.get('annotation_files',0)}")

        ontology_repo  = OntologyRepository(repo=repo)
        neo4j_texts    = ontology_repo.collect_ontology_texts()

        graph        = load_ontology_from_json(GRAPH_JSON)
        graph_texts  = build_ontology_texts(graph)

        ann_files    = find_annotation_files(ANNOTATION_DIR)
        annotations  = load_text_annotations(ann_files)
        node_names   = {n.uri: n.name_ru for n in graph.nodes}
        ann_texts    = build_texts_from_annotations(
            annotations, node_names=node_names, max_mentions=4)

        combined: dict = {}
        for t in neo4j_texts:
            combined[t["source_id"]] = t
        for t in graph_texts:
            combined[t["source_id"]] = t
        for t in ann_texts:
            sid = t["source_id"]
            if sid in combined:
                extra = ". ".join(t["text"].split(". ")[1:])
                if extra:
                    combined[sid]["text"] = combined[sid]["text"].rstrip(". ") + ". " + extra
            else:
                combined[sid] = t

        all_texts = list(combined.values())
        print(f"\n  Итого уникальных текстовых фрагментов: {len(all_texts)}")

        print("\n" + "=" * 60)
        print("ШАГ 3 (Фаза 1): Построение эмбеддингов")
        rag = RAGPipeline(
            embedding_service=embedding_service,
            search_service=SearchService(embedding_service),
            llm=StubLLM(),
            top_n=5, top_m=5,
        )
        n = rag.index_texts(all_texts, chunk_size=300, overlap=50)
        print(f"  Проиндексировано чанков: {n}")

        print("\n" + "=" * 60)
        print("ШАГ 4 (Фазы 2-3): Вопрос -> Поиск -> LLM")
        for q in [
            "Что такое Бетельгейзе и к какому типу звёзд она относится?",
            "Какие типы планет существуют?",
            "Какие телескопы использовались для наблюдений?",
        ]:
            _print_rag(rag.ask(q))
            print()


if __name__ == "__main__":
    print("=" * 60)
    example_cosine_similarity()
    print()
    print("=" * 60)
    example_chunk_embedding_and_search()
    print()
    example_import_and_rag()
