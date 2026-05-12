import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from json_ontology_loader import load_ontology_from_json, build_ontology_texts
from text_json_loader import (
    AnnotationCorpus,
    find_annotation_files,
    load_text_annotations,
)
from embedding_service import EmbeddingService
from llm_service import BaseLLM, StubLLM, TransformersLLM, DeepseekApiLLM
from rag_pipeline import RAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GRAPH_JSON = "graph.json"
ANNOTATION_DIR = "."
EMBEDDING_MODEL = "./paraphrase-multilingual-mpnet-base-v2"

TOP_N = 5
TOP_M = 5
K_SENTENCES = 2
L_FRAGMENTS = 3


def download_model(
    repo_id: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    local_dir: str = EMBEDDING_MODEL,
) -> None:
    from huggingface_hub import snapshot_download
    logger.info("Скачивание модели %s ...", repo_id)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    logger.info("Модель сохранена в %s", local_dir)


def create_llm(backend: str) -> BaseLLM:
    if backend == "stub":
        logger.info("LLM: StubLLM (заглушка).")
        return StubLLM()

    if backend == "transformers":
        model = os.getenv("TRANSFORMERS_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        logger.info("LLM: TransformersLLM — %s", model)
        return TransformersLLM(model_name=model)

    if backend == "deepseek":
        api_key = os.getenv("TIMEWEB_AI_TOKEN", "")
        if not api_key:
            raise RuntimeError("Не задан TIMEWEB_AI_TOKEN")
        base_url = (
            ""
        )
        model = "gpt-4.1"
        logger.info("LLM: Timeweb Agent API — %s (%s)", model, base_url)
        return DeepseekApiLLM(api_key=api_key, base_url=base_url, model=model)

    raise ValueError(f"Неизвестный LLM бэкенд: '{backend}'.")


def load_data() -> Tuple[List[Dict[str, Any]], AnnotationCorpus]:
    ontology_path = Path(GRAPH_JSON)
    if not ontology_path.exists():
        logger.error("Файл онтологии не найден: %s", ontology_path)
        sys.exit(1)

    logger.info("Загрузка онтологии из %s ...", ontology_path)
    graph = load_ontology_from_json(ontology_path)
    logger.info(
        "Онтология: %d узлов, %d свойств, %d связей",
        len(graph.nodes), len(graph.properties), len(graph.arcs),
    )

    node_names = {n.uri: n.name_ru for n in graph.nodes}
    ontology_texts = build_ontology_texts(graph)
    for t in ontology_texts:
        t["name_ru"] = node_names.get(t.get("source_id", ""), "")
    logger.info("Текстов сущностей онтологии: %d", len(ontology_texts))

    ann_files = find_annotation_files(ANNOTATION_DIR)
    logger.info("Найдено файлов аннотаций: %d", len(ann_files))

    if ann_files:
        corpus = load_text_annotations(ann_files)
    else:
        logger.warning("Файлы аннотаций не найдены — фрагменты доступны не будут.")
        corpus = AnnotationCorpus()

    logger.info(
        "Корпус: %d документов, %d сущностей с упоминаниями.",
        len(corpus.documents), len(corpus.occurrences),
    )

    return ontology_texts, corpus


def print_result(result: Dict[str, Any]) -> None:
    print()
    print("─" * 70)
    print(f"Вопрос: {result['question']}")

    print("\n[Фаза 2] Промежуточный ответ:")
    print(result["intermediate_answer"])

    print("\n[Фаза 2] Сущности N:")
    for e in result["phase2_entities"]:
        print(f"  score={e['score']:.4f} | {e['name']}")

    print("\n[Фаза 3] Сущности M (по промежуточному ответу):")
    for e in result["phase3_entities"]:
        print(f"  score={e['score']:.4f} | {e['name']}")

    print("\n[Фаза 3] Объединённые сущности (N+M):")
    for e in result["combined_entities"]:
        print(f"  score={e['score']:.4f} | {e['name']}")

    print("\n[Фаза 3] Отобранные фрагменты:")
    for f in result["fragments"]:
        snippet = f["text"][:120].replace("\n", " ")
        print(f"  [{f['entity']}] score={f['score']:.4f} | {snippet}...")

    print("\n[Финал] Ответ:")
    print(result["final_answer"])

    print("─" * 70)


def run_interactive(rag: RAGPipeline, demo_questions: List[str]) -> None:
    print("\n" + "=" * 70)
    print("RAG-система готова.")
    print("Введите вопрос или номер демо-вопроса (выход — 'q' / Ctrl+C).")
    print()
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    print("=" * 70)

    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("q", "quit", "выход", "exit"):
            break

        if question.isdigit() and 1 <= int(question) <= len(demo_questions):
            question = demo_questions[int(question) - 1]
            print(f"→ {question}")

        try:
            result = rag.ask(question)
            print_result(result)
        except Exception as exc:
            logger.error("Ошибка при обработке вопроса: %s", exc)

    print("\nЗавершение работы.")


def main() -> None:
    print("=" * 70)
    logger.info("Запуск RAG-системы по онтологии")
    print("=" * 70)

    logger.info("ФАЗА 1: Подготовка данных")
    ontology_texts, corpus = load_data()

    embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
    llm = create_llm("deepseek")

    rag = RAGPipeline(
        embedding_service=embedding_service,
        llm=llm,
        annotation_corpus=corpus,
        top_n=TOP_N,
        top_m=TOP_M,
        sentence_window_k=K_SENTENCES,
        fragments_per_entity_l=L_FRAGMENTS,
    )

    n_indexed = rag.index_ontology_entities(ontology_texts)
    logger.info("Проиндексировано сущностей онтологии: %d.", n_indexed)

    logger.info("ФАЗЫ 2-3: Вопрос-Ответ")

    demo_questions = [
        "Какие типы планет существуют?",
        "Что такое экзопланета?",
        "Какие телескопы используются для наблюдений?",
        "Что такое Бетельгейзе и к какому типу звёзд она относится?",
        "Чем нейтронная звезда отличается от белого карлика?",
    ]

    run_interactive(rag, demo_questions)


if __name__ == "__main__":
    main()