import argparse
import asyncio
import sys
from app.services.ocr_service import OCRService
from app.services.cleaner_service import CleanerService
from app.services.vector_service import VectorService

def run_ocr(args):
    print("Запуск OCR...")
    service = OCRService()
    processed = service.process_directory()
    print(f"Обработано файлов: {len(processed)}")
    for f in processed:
        print(f" - {f}")

def run_clean(args):
    print("Запуск очистки...")
    service = CleanerService()
    cleaned = service.process_directory()
    print(f"Очищено файлов: {len(cleaned)}")
    for f in cleaned:
        print(f" - {f}")

def run_index(args):
    print("Запуск индексации...")
    service = VectorService()
    count = service.index_directory()
    print(f"Проиндексировано документов: {count}")

def run_search(args):
    if not args.query:
        print("Ошибка: укажите запрос через --query")
        return
    
    print(f"Поиск: '{args.query}'...")
    service = VectorService()
    results = service.search(args.query, limit=args.limit)
    
    if not results:
        print("Ничего не найдено.")
    
    for res in results:
        print(f"[{res['score']:.4f}] {res['filename']}")
        print(f"   {res['text'][:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Diplom OCR & RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # OCR command
    ocr_parser = subparsers.add_parser("ocr", help="Запустить OCR")
    ocr_parser.set_defaults(func=run_ocr)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Запустить очистку XML")
    clean_parser.set_defaults(func=run_clean)

    # Index command
    index_parser = subparsers.add_parser("index", help="Индексировать документы в Qdrant")
    index_parser.set_defaults(func=run_index)

    # Search command
    search_parser = subparsers.add_parser("search", help="Поиск по документам")
    search_parser.add_argument("--query", type=str, required=True, help="Текст запроса")
    search_parser.add_argument("--limit", type=int, default=5, help="Количество результатов")
    search_parser.set_defaults(func=run_search)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
