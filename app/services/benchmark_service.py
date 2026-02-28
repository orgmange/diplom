from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client.http import models

from app.core.config import settings
from app.services.cleaner_service import CleanerService
from app.services.ocr_service import OCRService
from app.services.vector_service import VectorService


@dataclass
class BenchmarkItem:
    filename: str
    expected_type: Optional[str]
    predicted_type: Optional[str]
    predicted_filename: Optional[str]
    score: Optional[float]
    is_correct: bool
    alternatives: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "expected_type": self.expected_type,
            "predicted_type": self.predicted_type,
            "predicted_filename": self.predicted_filename,
            "score": self.score,
            "is_correct": self.is_correct,
            "alternatives": self.alternatives,
        }


@dataclass
class BenchmarkTotals:
    total: int
    correct: int
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
        }


class BenchmarkService:
    """Сервис для полного теста retrieval по разным embedding-моделям."""

    def __init__(
        self,
        vector_service: VectorService,
        ocr_service: Optional[OCRService] = None,
        cleaner_service: Optional[CleanerService] = None,
    ):
        self.vector_service = vector_service
        self.ocr_service = ocr_service or OCRService()
        self.cleaner_service = cleaner_service or CleanerService()
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self._stop_requested = False

    def stop(self):
        """Прерывает выполнение бенчмарка."""
        self._stop_requested = True

    def _iter_doc_dirs(self, docs_dir: Path) -> List[Path]:
        if not docs_dir.exists():
            return []
        return sorted(path for path in docs_dir.iterdir() if path.is_dir())

    def _prepare_test_corpus(self, docs_dir: Path) -> Dict[str, int]:
        prepared = 0
        prepared_clean = 0
        docs_dir.mkdir(parents=True, exist_ok=True)
        for doc_dir in self._iter_doc_dirs(docs_dir):
            image_dir = doc_dir / "image"
            xml_dir = doc_dir / "xml"
            clean_dir = doc_dir / "clean"
            xml_dir.mkdir(parents=True, exist_ok=True)
            clean_dir.mkdir(parents=True, exist_ok=True)
            if not image_dir.exists():
                continue
            images = sorted(
                path for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.image_extensions
            )
            for image_path in images:
                if self._stop_requested:
                    break
                xml_path = xml_dir / f"{image_path.name}-xml"
                clean_path = clean_dir / f"{image_path.name}-clean"
                if not xml_path.exists():
                    try:
                        task_id = self.ocr_service.create_task(image_path)
                        status = self.ocr_service.wait_for_task(task_id)
                        if status == "success":
                            content = self.ocr_service.fetch_result(task_id)
                            if content:
                                xml_path.write_bytes(content)
                                prepared += 1
                    except Exception:
                        continue
                if xml_path.exists() and not clean_path.exists():
                    try:
                        xml_bytes = xml_path.read_bytes()
                        clean_text = self.cleaner_service.parse_xml_bytes(xml_bytes)
                        if clean_text:
                            clean_path.write_text(clean_text, encoding="utf-8")
                            prepared_clean += 1
                    except Exception:
                        continue
        return {"prepared_xml": prepared, "prepared_clean": prepared_clean}

    def _detect_doc_type(self, filename: str) -> Optional[str]:
        lowered = filename.lower()
        rules = (
            ("passport", "passport"),
            ("паспорт", "passport"),
            ("prava", "driver_license"),
            ("права", "driver_license"),
            ("driver", "driver_license"),
            ("snils", "snils"),
            ("снилс", "snils"),
            ("svid", "birth_certificate"),
            ("свид", "birth_certificate"),
            ("birth", "birth_certificate"),
        )
        for token, doc_type in rules:
            if token in lowered:
                return doc_type
        return None

    def _evaluate_from_docs(
        self,
        docs_dir: Path,
        mode_dir: str,
        embedding_model: str,
        is_cleaned: bool,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="is_cleaned", match=models.MatchValue(value=is_cleaned)),
                models.FieldCondition(key="is_example", match=models.MatchValue(value=False)),
            ]
        )
        items: List[BenchmarkItem] = []
        for doc_dir in self._iter_doc_dirs(docs_dir):
            expected_type = doc_dir.name
            source_dir = doc_dir / mode_dir
            if not source_dir.exists():
                continue
            files = sorted(path for path in source_dir.iterdir() if path.is_file())
            for path in files:
                if self._stop_requested:
                    break
                query = path.read_text(encoding="utf-8").strip()
                item_name = f"{expected_type}/{path.name}"
                if not query:
                    items.append(
                        BenchmarkItem(
                            filename=item_name,
                            expected_type=expected_type,
                            predicted_type=None,
                            predicted_filename=None,
                            score=None,
                            is_correct=False,
                        )
                    )
                    continue
                results = self.vector_service.search(
                    query=query,
                    limit=3,
                    embedding_model=embedding_model,
                    query_filter=query_filter,
                    collection_name=collection_name
                )
                if not results:
                    items.append(
                        BenchmarkItem(
                            filename=item_name,
                            expected_type=expected_type,
                            predicted_type=None,
                            predicted_filename=None,
                            score=None,
                            is_correct=False,
                        )
                    )
                    continue
                top = results[0]
                predicted_filename = top.get("filename")
                predicted_type = top.get("doc_type") or self._detect_doc_type(predicted_filename or "")
                score = top.get("score")
                
                # Собираем альтернативы (топ-3)
                alternatives = []
                for res in results[1:]:
                    alt_filename = res.get("filename")
                    alternatives.append({
                        "filename": alt_filename,
                        "type": res.get("doc_type") or self._detect_doc_type(alt_filename or ""),
                        "score": res.get("score")
                    })

                items.append(
                    BenchmarkItem(
                        filename=item_name,
                        expected_type=expected_type,
                        predicted_type=predicted_type,
                        predicted_filename=predicted_filename,
                        score=score,
                        is_correct=expected_type == predicted_type if expected_type else False,
                        alternatives=alternatives
                    )
                )
        correct = sum(1 for item in items if item.is_correct)
        total = len(items)
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "items": [item.to_dict() for item in items],
        }

    def _combine_totals(self, raw_report: Dict[str, Any], clean_report: Dict[str, Any]) -> BenchmarkTotals:
        total = int(raw_report.get("total", 0)) + int(clean_report.get("total", 0))
        correct = int(raw_report.get("correct", 0)) + int(clean_report.get("correct", 0))
        accuracy = round(correct / total, 4) if total else 0.0
        return BenchmarkTotals(total=total, correct=correct, accuracy=accuracy)

    def run(self, embedding_model: str) -> Dict[str, Any]:
        """Запускает цикл reset, index и проверку raw/clean тестов в отдельной коллекции."""
        self._stop_requested = False
        template_dir = settings.OCR_DIR
        docs_dir = settings.DOCS_DIR
        benchmark_collection = "benchmark_documents"
        
        # Initialize default response structure
        prepared = {"prepared_xml": 0, "prepared_clean": 0}
        indexed = {"raw_count": 0, "clean_count": 0, "total_count": 0, "raw_files": [], "clean_files": []}
        raw_report = {"total": 0, "correct": 0, "accuracy": 0.0, "items": []}
        clean_report = {"total": 0, "correct": 0, "accuracy": 0.0, "items": []}
        overall = {"total": 0, "correct": 0, "accuracy": 0.0}

        prepared = self._prepare_test_corpus(docs_dir)
        if self._stop_requested:
            return self._format_run_result(embedding_model, prepared, indexed, raw_report, clean_report)

        vector_size = self.vector_service.get_embedding_size(embedding_model=embedding_model)
        
        # Используем отдельную коллекцию для бенчмарка
        self.vector_service.reset_collection(vector_size=vector_size, collection_name=benchmark_collection)
        
        if self._stop_requested:
            return self._format_run_result(embedding_model, prepared, indexed, raw_report, clean_report)

        indexed_raw = self.vector_service.index_templates_by_mode(
            is_cleaned=False,
            directory=template_dir,
            embedding_model=embedding_model,
            collection_name=benchmark_collection
        )
        indexed["raw_count"] = len(indexed_raw)
        indexed["raw_files"] = indexed_raw
        indexed["total_count"] = indexed["raw_count"] + indexed["clean_count"]

        if self._stop_requested:
            return self._format_run_result(embedding_model, prepared, indexed, raw_report, clean_report)

        indexed_clean = self.vector_service.index_templates_by_mode(
            is_cleaned=True,
            directory=template_dir,
            embedding_model=embedding_model,
            collection_name=benchmark_collection
        )
        indexed["clean_count"] = len(indexed_clean)
        indexed["clean_files"] = indexed_clean
        indexed["total_count"] = indexed["raw_count"] + indexed["clean_count"]

        if self._stop_requested:
            return self._format_run_result(embedding_model, prepared, indexed, raw_report, clean_report)

        raw_report = self._evaluate_from_docs(
            docs_dir=docs_dir,
            mode_dir="xml",
            embedding_model=embedding_model,
            is_cleaned=False,
            collection_name=benchmark_collection
        )
        if self._stop_requested:
            return self._format_run_result(embedding_model, prepared, indexed, raw_report, clean_report)

        clean_report = self._evaluate_from_docs(
            docs_dir=docs_dir,
            mode_dir="clean",
            embedding_model=embedding_model,
            is_cleaned=True,
            collection_name=benchmark_collection
        )
        
        # Сохраняем модель как текущую после бенчмарка
        if hasattr(self.vector_service, "_save_state"):
            self.vector_service._save_state(embedding_model)
            
        return self._format_run_result(embedding_model, prepared, indexed, raw_report, clean_report)

    def _format_run_result(self, embedding_model, prepared, indexed, raw_report, clean_report):
        overall = self._combine_totals(raw_report=raw_report, clean_report=clean_report)
        return {
            "embedding_model": embedding_model,
            "prepared": prepared,
            "indexed": indexed,
            "overall": overall.to_dict(),
            "raw_tests": raw_report,
            "clean_tests": clean_report,
        }
