import os
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Any

from app.core.config import settings
from app.services.vector_service import VectorService
from app.services.structuring_service import StructuringService

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate-references")

def generate_references(model_name: str = "qwen3:8b", embedding_model: str = "embeddinggemma"):
    """
    Оболочка для запуска асинхронной функции.
    """
    asyncio.run(async_generate_references(model_name, embedding_model))

async def async_generate_references(model_name: str, embedding_model: str):
    """
    Проходит по всем очищенным файлам в data/docs и генерирует эталонные JSON.
    """
    vector_service = VectorService()
    structuring_service = StructuringService(vector_service)
    
    docs_dir = settings.DOCS_DIR
    # Используем data/references как указал пользователь
    ref_dir = settings.DATA_DIR / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    if not docs_dir.exists():
        logger.error(f"Docs directory {docs_dir} does not exist.")
        return

    # Собираем список всех файлов для обработки
    files_to_process = []
    for doc_type_dir in sorted(docs_dir.iterdir()):
        if not doc_type_dir.is_dir():
            continue
        clean_dir = doc_type_dir / "clean"
        xml_dir = doc_type_dir / "xml"
        if not clean_dir.exists():
            continue
        for clean_file in sorted(clean_dir.iterdir()):
            if clean_file.is_file():
                files_to_process.append((clean_file, xml_dir, doc_type_dir.name))

    logger.info(f"Generating references for {len(files_to_process)} files using {model_name}...")

    for clean_file, xml_dir, doc_type in files_to_process:
        # Имя для референса (например, passport2.jpg-reference.json)
        image_name = clean_file.name.replace("-clean", "")
        ref_filename = f"{image_name}-reference.json"
        ref_path = ref_dir / ref_filename
        
        # Если файл уже существует, можно пропустить
        if ref_path.exists():
            logger.info(f"Reference for {image_name} already exists, skipping.")
            continue

        # Загрузка очищенного текста
        cleaned_text = clean_file.read_text(encoding="utf-8").strip()
        
        # Поиск соответствующего сырого текста
        raw_filename = clean_file.name.replace("-clean", "-xml")
        raw_file = xml_dir / raw_filename
        raw_text = cleaned_text
        if raw_file.exists():
            raw_text = raw_file.read_text(encoding="utf-8").strip()

        logger.info(f"Generating JSON for {image_name} ({doc_type})...")
        
        try:
            start_time = time.time()
            struct_data = await structuring_service.structure(
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                model_name=model_name,
                embedding_model=embedding_model,
                expected_type=doc_type
            )
            duration = time.time() - start_time
            
            result_json = struct_data["result"]
            
            # Сохраняем результат
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  Done in {duration:.2f}s -> {ref_filename}")
            
        except Exception as e:
            logger.error(f"  Failed to generate reference for {image_name}: {e}")

if __name__ == "__main__":
    # Используем имена моделей как просил пользователь
    # ВНИМАНИЕ: Если модели qwen3:8b или embeddinggemma не установлены, это упадет.
    # Возможно вы имели в виду qwen2.5:latest или nomic-embed-text?
    # Но ставлю как просили.
    generate_references(
        model_name="qwen3:8b", 
        embedding_model="embeddinggemma"
    )
