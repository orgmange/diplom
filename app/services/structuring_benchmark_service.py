import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.structuring_service import StructuringService
from qdrant_client.http import models

logger = logging.getLogger("diplom")

@dataclass
class StructuringBenchmarkItem:
    filename: str
    expected_type: str
    detected_type: str
    is_type_correct: bool
    processing_time: float
    accuracy: float
    is_reference_found: bool
    result_json: Dict[str, Any]
    reference_json: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "expected_type": self.expected_type,
            "detected_type": self.detected_type,
            "is_type_correct": self.is_type_correct,
            "processing_time": round(self.processing_time, 3),
            "accuracy": round(self.accuracy, 4),
            "is_reference_found": self.is_reference_found,
            "result_json": self.result_json,
            "reference_json": self.reference_json,
        }

@dataclass
class StructuringBenchmarkReport:
    model_name: str
    embedding_model: str
    total_files: int
    files_with_reference: int
    correct_templates_count: int
    template_accuracy: float
    avg_processing_time: float
    avg_accuracy: float
    items: List[StructuringBenchmarkItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "total_files": self.total_files,
            "files_with_reference": self.files_with_reference,
            "correct_templates_count": self.correct_templates_count,
            "template_accuracy": round(self.template_accuracy, 4),
            "avg_processing_time": round(self.avg_processing_time, 3),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "items": [item.to_dict() for item in self.items],
        }

class StructuringBenchmarkService:
    """
    Сервис для тестирования качества структурирования данных через LLM.
    """
    def __init__(self, structuring_service: StructuringService):
        self.structuring_service = structuring_service

    def _calculate_accuracy(self, result: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """Рекурсивно сравнивает поля в результате и эталоне, возвращая долю совпадений."""
        if not reference:
            return 0.0

        matches = 0
        total_keys = 0

        def _normalize(val: Any) -> str:
            if val is None:
                return ""
            return str(val).strip().upper()

        for key, expected_val in reference.items():
            total_keys += 1
            actual_val = result.get(key)

            if isinstance(expected_val, dict) and isinstance(actual_val, dict):
                # Для вложенных словарей (например, mrz)
                nested_accuracy = self._calculate_accuracy(actual_val, expected_val)
                if nested_accuracy == 1.0:
                    matches += 1
            else:
                if _normalize(actual_val) == _normalize(expected_val):
                    matches += 1

        return matches / total_keys if total_keys > 0 else 1.0

    def run(self, model_name: str, embedding_model: Optional[str] = None) -> StructuringBenchmarkReport:
        """Запускает прогон всех доступных документов через выбранную LLM модель."""
        docs_dir = settings.DOCS_DIR
        ref_dir = settings.BENCHMARK_REF_DIR
        
        # Предварительная проверка наличия примеров
        try:
            example_count_res = self.structuring_service.vector_service.client.count(
                collection_name=settings.COLLECTION_NAME,
                exact=True,
                count_filter=models.Filter(must=[models.FieldCondition(key="is_example", match=models.MatchValue(value=True))])
            )
            if example_count_res.count == 0:
                logger.warning("No examples found in collection. Auto-indexing examples for RAG...")
                self.structuring_service.vector_service.index_examples(embedding_model=embedding_model)
        except Exception as e:
            logger.error(f"Error checking example count: {e}")

        items: List[StructuringBenchmarkItem] = []
        
        if not docs_dir.exists():
            return StructuringBenchmarkReport(
                model_name=model_name,
                embedding_model=embedding_model or "default",
                total_files=0,
                files_with_reference=0,
                avg_processing_time=0.0,
                avg_accuracy=0.0,
                items=[]
            )

        # Сбор списка всех файлов для обработки
        files_to_process: List[Tuple[Path, Path, str]] = []
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

        logger.info(f"Starting structuring benchmark for {len(files_to_process)} files using model {model_name}")

        for clean_file, xml_dir, doc_type in files_to_process:
            # Загрузка очищенного текста
            cleaned_text = clean_file.read_text(encoding="utf-8").strip()
            
            # Поиск соответствующего сырого текста
            raw_filename = clean_file.name.replace("-clean", "-xml")
            raw_file = xml_dir / raw_filename
            raw_text = cleaned_text
            if raw_file.exists():
                raw_text = raw_file.read_text(encoding="utf-8").strip()

            logger.info(f"Processing {clean_file.name} (real type: {doc_type}, but not giving hints to LLM)...")
            # Замер времени структурирования
            start_time = time.time()
            struct_data = self.structuring_service.structure(
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                model_name=model_name,
                embedding_model=embedding_model,
                expected_type=doc_type
            )
            duration = time.time() - start_time
            
            result_json = struct_data["result"]
            detected_type = struct_data["doc_type"]
            is_type_correct = (detected_type == doc_type)

            # Поиск эталонного JSON
            image_name = clean_file.name.replace("-clean", "")
            ref_file = ref_dir / f"{image_name}-reference.json"
            
            accuracy = 0.0
            reference_json = None
            is_reference_found = False
            
            if ref_file.exists():
                try:
                    reference_json = json.loads(ref_file.read_text(encoding="utf-8"))
                    accuracy = self._calculate_accuracy(result_json, reference_json)
                    is_reference_found = True
                except Exception as e:
                    logger.error(f"Error reading reference file {ref_file}: {e}")

            items.append(StructuringBenchmarkItem(
                filename=image_name,
                expected_type=doc_type,
                detected_type=detected_type or "unknown",
                is_type_correct=is_type_correct,
                processing_time=duration,
                accuracy=accuracy,
                is_reference_found=is_reference_found,
                result_json=result_json,
                reference_json=reference_json
            ))

        total_files = len(items)
        files_with_reference = sum(1 for item in items if item.is_reference_found)
        correct_templates_count = sum(1 for item in items if item.is_type_correct)
        template_accuracy = correct_templates_count / total_files if total_files > 0 else 0.0
        
        avg_time = sum(item.processing_time for item in items) / total_files if total_files > 0 else 0.0
        
        acc_sum = sum(item.accuracy for item in items if item.is_reference_found)
        avg_accuracy = acc_sum / files_with_reference if files_with_reference > 0 else 0.0
        
        return StructuringBenchmarkReport(
            model_name=model_name,
            embedding_model=embedding_model or "default",
            total_files=total_files,
            files_with_reference=files_with_reference,
            correct_templates_count=correct_templates_count,
            template_accuracy=template_accuracy,
            avg_processing_time=avg_time,
            avg_accuracy=avg_accuracy,
            items=items
        )
