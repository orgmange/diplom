import json
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from app.core.config import settings
from app.services.structuring_service import StructuringService
from qdrant_client.http import models

logger = logging.getLogger("diplom")

@dataclass
class FieldMetrics:
    """Метрики отдельного поля."""
    field_name: str
    expected: str
    actual: str
    is_exact_match: bool
    cer: float
    fuzzy_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "expected": self.expected,
            "actual": self.actual,
            "is_exact_match": self.is_exact_match,
            "cer": round(self.cer, 4),
            "fuzzy_score": round(self.fuzzy_score, 4),
        }

@dataclass
class StructuringBenchmarkItem:
    filename: str
    expected_type: str
    detected_type: str
    is_type_correct: bool
    processing_time: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_cer: float
    avg_fuzzy_score: float
    is_reference_found: bool
    result_json: Dict[str, Any]
    reference_json: Optional[Dict[str, Any]] = None
    field_metrics: List[FieldMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "expected_type": self.expected_type,
            "detected_type": self.detected_type,
            "is_type_correct": self.is_type_correct,
            "processing_time": round(self.processing_time, 3),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_cer": round(self.avg_cer, 4),
            "avg_fuzzy_score": round(self.avg_fuzzy_score, 4),
            "is_reference_found": self.is_reference_found,
            "result_json": self.result_json,
            "reference_json": self.reference_json,
            "field_metrics": [fm.to_dict() for fm in self.field_metrics],
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
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_cer: float
    avg_fuzzy_score: float
    items: List[StructuringBenchmarkItem]
    # Strict metrics (excluding incorrectly classified types)
    avg_precision_strict: float = 0.0
    avg_recall_strict: float = 0.0
    avg_f1_strict: float = 0.0
    avg_fuzzy_strict: float = 0.0
    temperature: float = 0.0
    num_ctx: int = 16384
    timeout: int = 60
    structured_output: bool = True
    use_rag: bool = True

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
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_cer": round(self.avg_cer, 4),
            "avg_fuzzy_score": round(self.avg_fuzzy_score, 4),
            "avg_precision_strict": round(self.avg_precision_strict, 4),
            "avg_recall_strict": round(self.avg_recall_strict, 4),
            "avg_f1_strict": round(self.avg_f1_strict, 4),
            "avg_fuzzy_strict": round(self.avg_fuzzy_strict, 4),
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "timeout": self.timeout,
            "structured_output": self.structured_output,
            "use_rag": self.use_rag,
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

    @staticmethod
    def _normalize(val: Any) -> str:
        """Нормализует значение: убирает регистр, пробелы и знаки препинания."""
        if val is None:
            return ""
        text = str(val).strip().upper()
        text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
        return text

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """Рекурсивно разворачивает вложенные словари в плоский формат."""
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(StructuringBenchmarkService._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def _calculate_cer(actual: str, expected: str) -> float:
        """Вычисляет Character Error Rate (расстояние Левенштейна / длина эталона)."""
        if not expected:
            return 0.0 if not actual else 1.0
        from rapidfuzz.distance import Levenshtein
        distance = Levenshtein.distance(actual, expected)
        return distance / len(expected)

    @staticmethod
    def _calculate_fuzzy_score(actual: str, expected: str) -> float:
        """Вычисляет нечёткое совпадение строк через token_sort_ratio (0-100 -> 0-1)."""
        if not expected and not actual:
            return 1.0
        return fuzz.token_sort_ratio(actual, expected) / 100.0

    def _calculate_field_metrics(
        self, result: Dict[str, Any], reference: Dict[str, Any]
    ) -> Tuple[float, float, float, float, float, float, List[FieldMetrics]]:
        """Вычисляет все метрики: accuracy, precision, recall, f1, cer, fuzzy_score и детали по полям."""
        if not reference:
            return 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, []

        flat_ref = self._flatten_dict(reference)
        flat_res = self._flatten_dict(result)

        ref_keys = set(flat_ref.keys())
        res_keys = set(flat_res.keys())

        true_positives = 0
        field_metrics_list: List[FieldMetrics] = []
        cer_values: List[float] = []
        fuzzy_values: List[float] = []

        for key in ref_keys:
            expected_norm = self._normalize(flat_ref[key])
            actual_norm = self._normalize(flat_res.get(key))

            is_exact = expected_norm == actual_norm
            if is_exact:
                true_positives += 1

            cer = self._calculate_cer(actual_norm, expected_norm)
            fuzzy_s = self._calculate_fuzzy_score(actual_norm, expected_norm)
            cer_values.append(cer)
            fuzzy_values.append(fuzzy_s)

            field_metrics_list.append(FieldMetrics(
                field_name=key,
                expected=str(flat_ref[key]),
                actual=str(flat_res.get(key, "")),
                is_exact_match=is_exact,
                cer=cer,
                fuzzy_score=fuzzy_s,
            ))

        accuracy = true_positives / len(ref_keys) if ref_keys else 1.0
        
        implicit_correct_empty = sum(1 for k in ref_keys if k not in res_keys and self._normalize(flat_ref[k]) == "")
        total_predictions = len(res_keys) + implicit_correct_empty
        
        precision = true_positives / total_predictions if total_predictions > 0 else 0.0
        recall = true_positives / len(ref_keys) if ref_keys else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        avg_cer = sum(cer_values) / len(cer_values) if cer_values else 0.0
        avg_fuzzy = sum(fuzzy_values) / len(fuzzy_values) if fuzzy_values else 0.0

        return accuracy, precision, recall, f1, avg_cer, avg_fuzzy, field_metrics_list

    def _calculate_accuracy(self, result: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """Обратная совместимость: возвращает долю точных совпадений полей."""
        accuracy, _, _, _, _, _, _ = self._calculate_field_metrics(result, reference)
        return accuracy

    async def run(
        self,
        model_name: str,
        embedding_model: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 16384,
        timeout: int = 60,
        structured_output: bool = True,
        use_rag: bool = True
    ) -> StructuringBenchmarkReport:
        """Запускает прогон всех доступных документов через выбранную LLM модель."""
        self._stop_requested = False
        docs_dir = settings.DOCS_DIR
        ref_dir = settings.BENCHMARK_REF_DIR
        
        # Синхронизация эмбеддинг-модели и переиндексация
        try:
            current_embed = self.structuring_service.vector_service._load_state()
            target_embed = embedding_model or current_embed
            
            # В новой системе мы просто вызываем reindex_all, который индексирует только примеры
            # Это гарантирует, что база всегда в актуальном состоянии для бенчмарка.
            logger.info(f"Re-indexing unified examples for benchmark (Target: {target_embed})")
            self.structuring_service.vector_service.reindex_all(embedding_model=target_embed)
        except Exception as e:
            logger.error(f"Error re-indexing: {e}")

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

            logger.info(
                f"[{i+1}/{len(files_to_process)}] Processing {clean_file.name} for model {model_name} "
                f"(T={temperature}, Ctx={num_ctx}, TO={timeout}, Struct={structured_output})..."
            )
            
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
                    on_chunk=update_local_stream,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    timeout=timeout,
                    structured_output=structured_output,
                    use_rag=use_rag,
                    expected_type=doc_type
                )
                
                prompt_size = struct_data.get("prompt_size", 0)
                if "error" in struct_data.get("result", {}):
                    consecutive_errors += 1
                    logger.warning(f"  Prompt: {prompt_size} chars. Error: {struct_data['result']['error']}")
                    logger.warning(f"  Consecutive error {consecutive_errors}/3 for model {model_name} on file {clean_file.name}")
                else:
                    # Даже если тип не совпал, это считается "успешным" ответом модели (она не зависла)
                    consecutive_errors = 0
                    logger.info(f"  Prompt: {prompt_size} chars. Success in {time.time() - start_time:.2f}s.")
                    
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
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            avg_cer = 1.0
            avg_fuzzy = 0.0
            reference_json = None
            is_reference_found = False
            field_metrics_list: List[FieldMetrics] = []
            
            if ref_file.exists():
                try:
                    reference_json = json.loads(ref_file.read_text(encoding="utf-8"))
                    accuracy, precision, recall, f1, avg_cer, avg_fuzzy, field_metrics_list = (
                        self._calculate_field_metrics(result_json, reference_json)
                    )
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
                precision=precision,
                recall=recall,
                f1=f1,
                avg_cer=avg_cer,
                avg_fuzzy_score=avg_fuzzy,
                is_reference_found=is_reference_found,
                result_json=result_json,
                reference_json=reference_json,
                field_metrics=field_metrics_list,
            ))
        
        # Завершаем прогресс для этой модели
        self._current_progress["processed_files"] = len(files_to_process)
        self._current_progress["current_file"] = "Завершено"

        total_files = len(items)
        files_with_reference = sum(1 for item in items if item.is_reference_found)
        correct_templates_count = sum(1 for item in items if item.is_type_correct)
        template_accuracy = correct_templates_count / total_files if total_files > 0 else 0.0
        
        avg_time = sum(item.processing_time for item in items) / total_files if total_files > 0 else 0.0
        
        ref_items = [item for item in items if item.is_reference_found]
        n_ref = len(ref_items) if ref_items else 1
        avg_accuracy = sum(item.accuracy for item in ref_items) / n_ref
        avg_precision = sum(item.precision for item in ref_items) / n_ref
        avg_recall = sum(item.recall for item in ref_items) / n_ref
        avg_f1 = sum(item.f1 for item in ref_items) / n_ref
        avg_cer = sum(item.avg_cer for item in ref_items) / n_ref
        avg_fuzzy = sum(item.avg_fuzzy_score for item in ref_items) / n_ref
        
        # Calculate strict metrics (only for correctly classified types)
        strict_items = [item for item in ref_items if item.is_type_correct]
        n_strict = len(strict_items)
        if n_strict > 0:
            avg_precision_strict = sum(item.precision for item in strict_items) / n_strict
            avg_recall_strict = sum(item.recall for item in strict_items) / n_strict
            avg_f1_strict = sum(item.f1 for item in strict_items) / n_strict
            avg_fuzzy_strict = sum(item.avg_fuzzy_score for item in strict_items) / n_strict
        else:
            avg_precision_strict = 0.0
            avg_recall_strict = 0.0
            avg_f1_strict = 0.0
            avg_fuzzy_strict = 0.0

        report = StructuringBenchmarkReport(
            model_name=model_name,
            embedding_model=embedding_model or "default",
            total_files=total_files,
            files_with_reference=files_with_reference,
            correct_templates_count=correct_templates_count,
            template_accuracy=template_accuracy,
            avg_processing_time=avg_time,
            avg_accuracy=avg_accuracy,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_cer=avg_cer,
            avg_fuzzy_score=avg_fuzzy,
            avg_precision_strict=avg_precision_strict,
            avg_recall_strict=avg_recall_strict,
            avg_f1_strict=avg_f1_strict,
            avg_fuzzy_strict=avg_fuzzy_strict,
            temperature=temperature,
            num_ctx=num_ctx,
            timeout=timeout,
            structured_output=structured_output,
            use_rag=use_rag,
            items=items
        )

        # Сохранение в историю
        self._save_report(report)
        return report

    async def run_multi(
        self,
        model_names: List[str],
        embedding_model: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 16383,
        timeout: int = 60,
        structured_output: bool = True,
        use_rag: bool = True
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
                    temperature=temperature,
                    num_ctx=num_ctx,
                    timeout=timeout,
                    structured_output=structured_output,
                    use_rag=use_rag
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
                        "precision": data.get("avg_precision"),
                        "recall": data.get("avg_recall"),
                        "f1": data.get("avg_f1"),
                        "f1_strict": data.get("avg_f1_strict"),
                        "fuzzy": data.get("avg_fuzzy_score"),
                        "template_accuracy": data.get("template_accuracy"),
                        "avg_processing_time": data.get("avg_processing_time"),
                        "use_rag": data.get("use_rag", True),
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

