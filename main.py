
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from json_ontology_loader import load_ontology_from_json, build_ontology_texts
from embedding_service import EmbeddingService
from search_service import SearchService
from llm_service import BaseLLM, StubLLM, TransformersLLM, DeepseekApiLLM
from rag_pipeline import RAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)



def download_model(
    repo_id: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    local_dir: str = "./paraphrase-multilingual-mpnet-base-v2",
):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    logger.info("Модель скачана в %s", local_dir)


def create_llm(args: argparse.Namespace) -> BaseLLM:
    if args.llm == "stub":
        logger.info("Используется StubLLM (заглушка).")
        return StubLLM()

    elif args.llm == "transformers":
        model = args.transformers_model or "meta-llama/Llama-3.1-8B-Instruct"
        logger.info("Используется TransformersLLM: %s", model)
        return TransformersLLM(model_name=model)

    elif args.llm == "deepseek":
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.error("Не задан API-ключ. Используйте --api-key или DEEPSEEK_API_KEY.")
            sys.exit(1)
        base_url = args.base_url or "https://api.deepseek.com"
        model = args.api_model or "deepseek-chat"
        logger.info("Используется DeepseekApiLLM: %s (%s)", model, base_url)
        return DeepseekApiLLM(api_key=api_key, base_url=base_url, model=model)

    else:
        raise ValueError(f"Неизвестный LLM бэкенд: {args.llm}")



def main():
    parser = argparse.ArgumentParser(description="RAG по онтологии")
    parser.add_argument(
        "--ontology", "-o",
        default="graph.json",
        help="Путь к JSON-файлу онтологии (по умолчанию: graph.json)",
    )
    parser.add_argument(
        "--llm",
        choices=["stub", "transformers", "deepseek"],
        default="stub",
        help="Бэкенд LLM (по умолчанию: stub — заглушка)",
    )
    parser.add_argument(
        "--embedding-model",
        default="./paraphrase-multilingual-mpnet-base-v2",
        help="Путь/имя модели для эмбеддингов",
    )
    parser.add_argument("--transformers-model", default=None, help="Модель для TransformersLLM")
    parser.add_argument("--api-key", default=None, help="API-ключ (для deepseek)")
    parser.add_argument("--base-url", default=None, help="Base URL для API")
    parser.add_argument("--api-model", default=None, help="Имя модели для API")
    parser.add_argument("--download-model", action="store_true", help="Скачать модель эмбеддингов")
    parser.add_argument("--top-n", type=int, default=5, help="Кол-во чанков на фазе 2 (N)")
    parser.add_argument("--top-m", type=int, default=5, help="Кол-во чанков на фазе 3 (M)")
    parser.add_argument("--chunk-size", type=int, default=300, help="Размер чанка (символы)")
    parser.add_argument("--overlap", type=int, default=50, help="Перекрытие чанков")

    args = parser.parse_args()

    if args.download_model:
        download_model()

    ontology_path = Path(args.ontology)
    if not ontology_path.exists():
        logger.error("Файл онтологии не найден: %s", ontology_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ФАЗА 1: Подготовка данных")
    logger.info("=" * 60)

    graph = load_ontology_from_json(ontology_path)
    logger.info(
        "Онтология загружена: %d узлов, %d свойств, %d связей",
        len(graph.nodes), len(graph.properties), len(graph.arcs),
    )

    ontology_texts = build_ontology_texts(graph)
    logger.info("Сгенерировано %d текстовых фрагментов.", len(ontology_texts))

    for t in ontology_texts[:3]:
        logger.info("  [%s] %s", t["source_type"], t["text"][:120] + "...")

    embedding_service = EmbeddingService(model_name=args.embedding_model)
    search_service = SearchService(embedding_service)
    llm = create_llm(args)

    rag = RAGPipeline(
        embedding_service=embedding_service,
        search_service=search_service,
        llm=llm,
        top_n=args.top_n,
        top_m=args.top_m,
    )

    num_chunks = rag.index_texts(
        ontology_texts,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    logger.info("Проиндексировано %d чанков. Готово к вопросам.", num_chunks)

    logger.info("=" * 60)
    logger.info("ФАЗЫ 2-3: Вопрос-Ответ")
    logger.info("=" * 60)

    demo_questions = [
        "Какие типы планет существуют?",
        "Что такое экзопланета?",
        "Какие телескопы используются для наблюдений?",
    ]

    print("\n" + "=" * 60)
    print("RAG-система готова. Введите вопрос (или 'выход' для завершения).")
    print("Примеры вопросов:")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    print("=" * 60)

    while True:
        try:
            question = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("выход", "exit", "quit"):
            break

        if question.isdigit() and 1 <= int(question) <= len(demo_questions):
            question = demo_questions[int(question) - 1]
            print(f"→ {question}")

        result = rag.ask(question)

        print("\n--- Промежуточный ответ (Фаза 2) ---")
        print(result["intermediate_answer"])
        print("\n--- Финальный ответ (Фаза 3) ---")
        print(result["final_answer"])

        print("\n--- Использованные чанки (Фаза 2, N) ---")
        for c in result["phase2_chunks"]:
            print(f"  score={c['score']:.4f} | {c['text'][:80]}...")

        print("\n--- Объединённые чанки (Фаза 3, N+M) ---")
        for c in result["phase3_chunks"]:
            print(f"  score={c['score']:.4f} | {c['text'][:80]}...")

    print("\nЗавершение работы.")


if __name__ == "__main__":
    main()
