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
        self._stop_requested = False
        self._skip_model_requested = False
        self._current_progress: Dict[str, Any] = {
            "is_running": False,
            "current_model": None,
            "processed_files": 0,
            "total_files": 0,
            "current_file": None,
            "current_stream": "",
            "models_to_run": [],
            "completed_models": []
        }

    def stop(self):
        """Прерывает выполнение бенчмарка."""
        self._stop_requested = True

    def skip_model(self):
        """Пропускает текущую модель в бенчмарке."""
        self._skip_model_requested = True

    def get_progress(self) -> Dict[str, Any]:
        """Возвращает текущий прогресс бенчмарка."""
        return self._current_progress

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

    async def run(self, model_name: str, embedding_model: Optional[str] = None) -> StructuringBenchmarkReport:
        """Запускает прогон всех доступных документов через выбранную LLM модель."""
        self._stop_requested = False
        docs_dir = settings.DOCS_DIR
        ref_dir = settings.BENCHMARK_REF_DIR
        
        # Синхронизация эмбеддинг-модели и переиндексация при необходимости
        try:
            current_embed = self.structuring_service.vector_service._load_state()
            target_embed = embedding_model or current_embed
            
            # Проверяем наличие примеров в основной коллекции
            example_count_res = self.structuring_service.vector_service.client.count(
                collection_name=settings.COLLECTION_NAME,
                exact=True,
                count_filter=models.Filter(must=[models.FieldCondition(key="is_example", match=models.MatchValue(value=True))])
            )
            
            if target_embed != current_embed or example_count_res.count == 0:
                logger.warning(f"Re-indexing examples for benchmark (Target: {target_embed}, Current: {current_embed}, Count: {example_count_res.count})")
                self.structuring_service.vector_service.reindex_all(embedding_model=target_embed)
        except Exception as e:
            logger.error(f"Error checking embedding state or re-indexing: {e}")

        items: List[StructuringBenchmarkItem] = []
        
        if not docs_dir.exists():
            return StructuringBenchmarkReport(
                model_name=model_name,
                embedding_model=embedding_model or "default",
                total_files=0,
                files_with_reference=0,
                correct_templates_count=0,
                template_accuracy=0.0,
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
        
        # Обновляем прогресс
        self._current_progress.update({
            "is_running": True,
            "current_model": model_name,
            "total_files": len(files_to_process),
            "processed_files": 0
        })

        consecutive_errors = 0
        for i, (clean_file, xml_dir, doc_type) in enumerate(files_to_process):
            if self._stop_requested:
                logger.info("Structuring benchmark stopped by user request.")
                break
            
            if self._skip_model_requested:
                logger.info(f"Skipping model {model_name} by user request.")
                break

            if consecutive_errors >= 3:
                logger.error(f"Skipping model {model_name} due to 3 consecutive errors.")
                break

            # Обновляем текущий файл в прогрессе
            self._current_progress["current_file"] = clean_file.name
            self._current_progress["processed_files"] = i
            self._current_progress["current_stream"] = ""

            async def update_local_stream(chunk_text: str):
                self._current_progress["current_stream"] += chunk_text

            # Загрузка очищенного текста
            cleaned_text = clean_file.read_text(encoding="utf-8").strip()
            
            # Поиск соответствующего сырого текста
            raw_filename = clean_file.name.replace("-clean", "-xml")
            raw_file = xml_dir / raw_filename
            raw_text = cleaned_text
            if raw_file.exists():
                raw_text = raw_file.read_text(encoding="utf-8").strip()

            logger.info(f"[{i+1}/{len(files_to_process)}] Processing {clean_file.name} for model {model_name}...")
            
            # Первый файл может загружаться долго (прогрев модели)
            if i == 0:
                logger.info("First file: allowing model to load into memory...")
                # Мы не меняем таймаут здесь (он уже 90с в клиенте), но логируем ожидание.
                # Можно добавить небольшой delay если нужно, но 90с должно хватить.

            # Замер времени структурирования
            start_time = time.time()
            try:
                struct_data = await self.structuring_service.structure(
                    raw_text=raw_text,
                    cleaned_text=cleaned_text,
                    model_name=model_name,
                    embedding_model=embedding_model,
                    on_chunk=update_local_stream
                )
                
                if "error" in struct_data.get("result", {}):
                    consecutive_errors += 1
                    logger.warning(f"Consecutive error {consecutive_errors}/3 for model {model_name} on file {clean_file.name}")
                else:
                    # Даже если тип не совпал, это считается "успешным" ответом модели (она не зависла)
                    consecutive_errors = 0
                    
            except Exception as e:
                logger.error(f"Critical error processing {clean_file.name}: {e}")
                consecutive_errors += 1
                struct_data = {
                    "result": {"error": str(e)},
                    "doc_type": "unknown"
                }

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
        
        # Завершаем прогресс для этой модели
        self._current_progress["processed_files"] = len(files_to_process)
        self._current_progress["current_file"] = "Завершено"

        total_files = len(items)
        files_with_reference = sum(1 for item in items if item.is_reference_found)
        correct_templates_count = sum(1 for item in items if item.is_type_correct)
        template_accuracy = correct_templates_count / total_files if total_files > 0 else 0.0
        
        avg_time = sum(item.processing_time for item in items) / total_files if total_files > 0 else 0.0
        
        acc_sum = sum(item.accuracy for item in items if item.is_reference_found)
        avg_accuracy = acc_sum / files_with_reference if files_with_reference > 0 else 0.0
        
        report = StructuringBenchmarkReport(
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

        # Сохранение в историю
        self._save_report(report)
        return report

    async def run_multi(
        self,
        model_names: List[str],
        embedding_model: Optional[str] = None,
    ) -> List[StructuringBenchmarkReport]:
        """Поочерёдно запускает бенчмарк структурирования для каждой модели из списка."""
        self._stop_requested = False
        reports: List[StructuringBenchmarkReport] = []
        
        self._current_progress.update({
            "is_running": True,
            "models_to_run": model_names,
            "completed_models": [],
            "current_model": None,
            "processed_files": 0,
            "total_files": 0
        })

        try:
            for model_name in model_names:
                if self._stop_requested:
                    logger.info("Multi-model benchmark stopped by user request.")
                    break

                self._skip_model_requested = False
                logger.info(f"Starting benchmark for model: {model_name}")
                report = await self.run(
                    model_name=model_name,
                    embedding_model=embedding_model,
                )
                reports.append(report)
                self._current_progress["completed_models"].append(model_name)
        finally:
            self._current_progress["is_running"] = False

        return reports

    def _save_report(self, report: StructuringBenchmarkReport):
        """Сохраняет отчет в JSON файл."""
        import datetime
        reports_dir = settings.BASE_DIR / "data" / "benchmark" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{report.model_name.replace(':', '_')}_{timestamp}.json"
        filepath = reports_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Benchmark report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")

    def list_reports(self) -> List[Dict[str, Any]]:
        """Возвращает список сохраненных отчетов (метаданные)."""
        reports_dir = settings.BASE_DIR / "data" / "benchmark" / "reports"
        if not reports_dir.exists():
            return []
            
        reports = []
        for file in sorted(reports_dir.glob("report_*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append({
                        "filename": file.name,
                        "model_name": data.get("model_name"),
                        "total_files": data.get("total_files"),
                        "accuracy": data.get("avg_accuracy"),
                        "template_accuracy": data.get("template_accuracy"),
                        "timestamp": file.stat().st_mtime
                    })
            except: continue
        return reports

    def get_report(self, filename: str) -> Optional[Dict[str, Any]]:
        """Загружает полный отчет из файла."""
        filepath = settings.BASE_DIR / "data" / "benchmark" / "reports" / filename
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return None

    def delete_report(self, filename: str) -> bool:
        """Удаляет конкретный отчет."""
        filepath = settings.BASE_DIR / "data" / "benchmark" / "reports" / filename
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted benchmark report: {filename}")
            return True
        return False

    def clear_reports(self) -> int:
        """Удаляет все отчеты."""
        reports_dir = settings.BASE_DIR / "data" / "benchmark" / "reports"
        if not reports_dir.exists():
            return 0
            
        count = 0
        for file in reports_dir.glob("report_*.json"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete report {file.name}: {e}")
        
        logger.info(f"Cleared {count} benchmark reports")
        return count

