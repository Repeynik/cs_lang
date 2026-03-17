

import logging
from cs_lang import EmbeddingService, EmbeddingSearchService, OntologyRepository, Neo4jRepository, OntologyEmbeddingPipeline
from huggingface_hub import snapshot_download


logging.basicConfig(level=logging.INFO)



def example_cosine_similarity():
    service = EmbeddingService(
        model_name="./paraphrase-multilingual-mpnet-base-v2"
    )
    text1 = "Кошка сидит на подоконнике"
    text2 = "На окне сидит кот"
    text3 = "Сервер не отвечает по SSH"

    embeddings = service.get_embeddings([text1, text2, text3])

    sim_1_2 = service.cos_compare(embeddings[0], embeddings[1])
    sim_1_3 = service.cos_compare(embeddings[0], embeddings[2])
    sim_2_3 = service.cos_compare(embeddings[1], embeddings[2])

    print("Пример косинусного сходства:")
    print(f"text1: {text1}")
    print(f"text2: {text2}")
    print(f"text3: {text3}")
    print()
    print(f"similarity(text1, text2) = {sim_1_2:.4f}")
    print(f"similarity(text1, text3) = {sim_1_3:.4f}")
    print(f"similarity(text2, text3) = {sim_2_3:.4f}")
    print()

    if sim_1_2 > sim_1_3:
        print("text1 и text2 семантически ближе, чем text1 и text3")
    else:
        print("text1 и text3 семантически ближе, чем text1 и text2")


def example_chunk_embedding_and_search():
    service = EmbeddingService(
        model_name="./paraphrase-multilingual-mpnet-base-v2"
    )
    search_service = EmbeddingSearchService(service)

    texts = [
        {
            "source_id": "class_1",
            "source_type": "Class",
            "text": "Класс: Сервер. Описание: Сервер предоставляет вычислительные ресурсы и сетевые сервисы.",
        },
        {
            "source_id": "class_2",
            "source_type": "Class",
            "text": "Класс: База данных. Описание: База данных хранит структурированную информацию и поддерживает запросы.",
        },
        {
            "source_id": "object_1",
            "source_type": "Object",
            "text": "Объект: web-01. Описание: Веб сервер production. Атрибуты: ip: 10.0.0.1; role: frontend",
        },
    ]

    chunks = service.get_chunks(
        texts=texts,
        chunk_size=120,
        overlap=20,
        min_chunk_length=20,
    )

    embedded_chunks = service.embed_chunks(chunks)

    query = "сервер для веб приложения"
    results = search_service.find_most_similar(
        query=query,
        stored_chunks=embedded_chunks,
        top_k=3,
    )

    print("Пример поиска похожих чанков:")
    print(f"query: {query}")
    print()

    for i, item in enumerate(results, start=1):
        score = item["score"]
        chunk = item["chunk"]["chunk"]
        print(f"{i}. score={score:.4f}")
        print(f"   source_id={chunk.source_id}")
        print(f"   source_type={chunk.source_type}")
        print(f"   text={chunk.text}")
        print()

def seed_demo_ontology():
    with Neo4jRepository(
        database_uri="bolt://127.0.0.1:7687",
        user="neo4j",
        password="123456789",
    ) as repo:
        ontology_repo = OntologyRepository(repo=repo)

        server_class = ontology_repo.create_class(
            name="Сервер",
            description="Узел, предоставляющий вычислительные и сетевые сервисы",
        )

        db_class = ontology_repo.create_class(
            name="База данных",
            description="Система хранения структурированных данных",
        )

        ontology_repo.add_class_attribue(
            class_uri=server_class.uri,
            attr_name="ip",
            datatype="string",
            description="IP адрес сервера",
        )

        ontology_repo.add_class_attribue(
            class_uri=server_class.uri,
            attr_name="role",
            datatype="string",
            description="Роль сервера",
        )

        ontology_repo.add_class_object_attribute(
            class_uri=server_class.uri,
            attr_name="uses_database",
            range_class_uri=db_class.uri,
            description="Сервер использует базу данных",
        )

        web_obj = ontology_repo.create_object(
            class_uri=server_class.uri,
            obj_params={
                "description": "Production web server",
                "title": "web-01",
            },
            data={
                "ip": "10.0.0.1",
                "role": "frontend",
            },
        )

        db_obj = ontology_repo.create_object(
            class_uri=db_class.uri,
            obj_params={
                "description": "Primary database",
                "title": "db-01",
            },
            data={},
        )

        ontology_repo.update_object(
            object_uri=web_obj.uri,
            data={
                "uses_database": db_obj.uri,
            },
        )

        print("Тестовая онтология создана")
        
def example_pipeline_with_neo4j():
    with Neo4jRepository(
        database_uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="123456789",
    ) as repo:
        ontology_repo = OntologyRepository(repo=repo)
        embedding_service = EmbeddingService(
        model_name="./paraphrase-multilingual-mpnet-base-v2"
    )
        pipeline = OntologyEmbeddingPipeline(
            ontology_repo=ontology_repo,
            embedding_service=embedding_service,
        )

        embedded_chunks = pipeline.build_embeddings_for_ontology(
            chunk_size=300,
            overlap=50,
        )

        print("Количество подготовленных чанков:", len(embedded_chunks))

        if len(embedded_chunks) >= 2:
            emb1 = embedded_chunks[0]["embedding"]
            emb2 = embedded_chunks[1]["embedding"]
            score = embedding_service.cos_compare(emb1, emb2)
            print("Cosine similarity между первыми двумя чанками:", round(score, 4))

        pipeline.save_embeddings_to_neo4j(
            chunk_size=300,
            overlap=50,
        )

        stored_chunks = ontology_repo.get_all_text_chunks()

        query = "класс для хранения данных"
        query_embedding = embedding_service.get_embeddings([query])[0]

        scored = []
        for chunk_data in stored_chunks:
            emb = chunk_data.get("embedding")
            if not emb:
                continue
            score = embedding_service.cos_compare(query_embedding, emb)
            scored.append({
                "score": score,
                "text": chunk_data.get("text", ""),
                "source_id": chunk_data.get("source_id", ""),
                "source_type": chunk_data.get("source_type", ""),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        print("Результаты поиска по уже сохраненным embedding в Neo4j:")
        for item in scored[:5]:
            print(f"score={item['score']:.4f} | {item['source_type']} | {item['source_id']}")
            print(item["text"])
            print()


if __name__ == "__main__":
    snapshot_download(
        repo_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        local_dir="./paraphrase-multilingual-mpnet-base-v2",
        local_dir_use_symlinks=False,
    )
    example_cosine_similarity()
    print("=" * 80)
    example_chunk_embedding_and_search()
    print("=" * 80)
    # seed_demo_ontology()
    example_pipeline_with_neo4j()
