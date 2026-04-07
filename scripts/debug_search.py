import asyncio
from app.services.vector_service import VectorService
from app.core.config import settings
from qdrant_client.http import models

async def test_discrepancy():
    vs = VectorService()
    file_path = settings.DATA_DIR / "docs/passport/clean/vqxkqp1sp23nwwshszdk.jpg-clean"
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        original_text = f.read().strip()
    
    # Схлопываем в одну строку (заменяем переносы на пробелы)
    collapsed_text = " ".join(original_text.split())
    
    model = "embeddinggemma"
    
    # Фильтр (без мета-фильтров по примеру/очистке)
    query_filter = None

    print(f"--- ТЕСТ МОДЕЛИ: {model} ---")
    
    print("\n1. ПОИСК С ОРИГИНАЛЬНЫМИ ПЕРЕНОСАМИ СТРОК:")
    results_orig = vs.search(original_text, limit=3, embedding_model=model, query_filter=query_filter)
    for r in results_orig:
        score = r.get('score', 0)
        print(f"   [{score*100:.1f}%] {r['filename']} (Type: {r['doc_type']})")

    print("\n2. ПОИСК В ОДНУ СТРОКУ (COLLAPSED):")
    results_coll = vs.search(collapsed_text, limit=3, embedding_model=model, query_filter=query_filter)
    for r in results_coll:
        score = r.get('score', 0)
        print(f"   [{score*100:.1f}%] {r['filename']} (Type: {r['doc_type']})")

if __name__ == "__main__":
    asyncio.run(test_discrepancy())
