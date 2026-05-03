import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client.http import models

from app.core.config import settings
from app.services.vector_service import VectorService

logger = logging.getLogger("diplom")

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

    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self._stop_requested = False

    def stop(self):
        """Прерывает выполнение бенчмарка."""
        self._stop_requested = True

    def _iter_doc_dirs(self, docs_dir: Path) -> List[Path]:
        if not docs_dir.exists():
            return []
        return sorted(path for path in docs_dir.iterdir() if path.is_dir())



    def _evaluate_from_docs(
        self,
        docs_dir: Path,
        mode_dir: str,
        embedding_model: str,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Оценивает точность поиска по документам (только чистый текст)."""
        items: List[BenchmarkItem] = []
        for doc_dir in self._iter_doc_dirs(docs_dir):
            expected_type = doc_dir.name
            # Мы теперь поддерживаем только clean режим в бенчмарках
            if mode_dir != "clean":
                continue
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
                predicted_type = top.get("doc_type")
                score = top.get("score")
                
                # Собираем альтернативы (топ-3)
                alternatives = []
                for res in results[1:]:
                    alt_filename = res.get("filename")
                    alternatives.append({
                        "filename": alt_filename,
                        "type": res.get("doc_type"),
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

    async def run(self, embedding_model: str) -> Dict[str, Any]:
        """Запускает бенчмарк поиска по единому хранилищу примеров."""
        self._stop_requested = False
        docs_dir = settings.DOCS_DIR
        benchmark_collection = "benchmark_documents"
        
        # Initialize default response structure
        indexed = {"total_count": 0, "files": []}
        clean_report = {"total": 0, "correct": 0, "accuracy": 0.0, "items": []}

        vector_size = self.vector_service.get_embedding_size(embedding_model=embedding_model)
        
        # Используем отдельную коллекцию для бенчмарка
        self.vector_service.reset_collection(vector_size=vector_size, collection_name=benchmark_collection)
        
        if self._stop_requested:
            return self._format_run_result(embedding_model, indexed, clean_report)

        # Индексируем только примеры (теперь это единый источник)
        indexed_files = await self.vector_service.index_examples(
            embedding_model=embedding_model,
            collection_name=benchmark_collection
        )
        indexed["total_count"] = len(indexed_files)
        indexed["files"] = indexed_files

        if self._stop_requested:
            return self._format_run_result(embedding_model, indexed, clean_report)

        # Оцениваем только по чистым документам
        clean_report = self._evaluate_from_docs(
            docs_dir=docs_dir,
            mode_dir="clean",
            embedding_model=embedding_model,
            collection_name=benchmark_collection
        )
        
        # Сохраняем модель как текущую после бенчмарка
        if hasattr(self.vector_service, "_save_state"):
            self.vector_service._save_state(embedding_model)
            
        result = self._format_run_result(embedding_model, indexed, clean_report)
        self._save_report(result)
        return result

    async def run_multi(self, embedding_models: List[str]) -> List[Dict[str, Any]]:
        """Поочерёдно запускает бенчмарк для списка моделей."""
        reports = []
        for model in embedding_models:
            if self._stop_requested:
                break
            logger.info(f"Starting retrieval benchmark for model: {model}")
            try:
                report = await self.run(model)
                reports.append(report)
            except Exception as e:
                logger.error(f"Error running benchmark for {model}: {e}")
                # Create a failure report
                failed_report = self._format_run_result(
                    model, 
                    {"total_count": 0, "files": []}, 
                    {"total": 0, "correct": 0, "accuracy": 0.0, "items": []},
                    error=str(e)
                )
                reports.append(failed_report)
        return reports

    def _format_run_result(self, embedding_model, indexed, clean_report, error=None):
        import time
        return {
            "embedding_model": embedding_model,
            "timestamp": time.time(),
            "indexed": indexed,
            "overall": {
                "total": clean_report.get("total", 0),
                "correct": clean_report.get("correct", 0),
                "accuracy": clean_report.get("accuracy", 0.0)
            },
            "clean_tests": clean_report,
            "error": error
        }

    def _save_report(self, report_data: Dict[str, Any]):
        """Сохраняет отчет в JSON файл."""
        import datetime
        import json
        reports_dir = settings.BASE_DIR / "data" / "benchmark" / "retrieval_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = report_data["embedding_model"].replace(":", "_").replace("/", "_")
        filename = f"retrieval_report_{model_safe}_{timestamp}.json"
        filepath = reports_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Retrieval benchmark report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save retrieval benchmark report: {e}")

    def list_reports(self) -> List[Dict[str, Any]]:
        """Возвращает список сохраненных отчетов (метаданные)."""
        import json
        reports_dir = settings.BASE_DIR / "data" / "benchmark" / "retrieval_reports"
        if not reports_dir.exists():
            return []
            
        reports = []
        for file in sorted(reports_dir.glob("retrieval_report_*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append({
                        "filename": file.name,
                        "embedding_model": data.get("embedding_model"),
                        "total_files": data.get("overall", {}).get("total"),
                        "accuracy": data.get("overall", {}).get("accuracy"),
                        "indexed_count": data.get("indexed", {}).get("total_count"),
                        "timestamp": data.get("timestamp") or file.stat().st_mtime
                    })
            except: continue
        return reports

    def get_report(self, filename: str) -> Optional[Dict[str, Any]]:
        """Загружает полный отчет из файла."""
        import json
        filepath = settings.BASE_DIR / "data" / "benchmark" / "retrieval_reports" / filename
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return None

    def delete_report(self, filename: str) -> bool:
        """Удаляет конкретный отчет."""
        filepath = settings.BASE_DIR / "data" / "benchmark" / "retrieval_reports" / filename
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted retrieval benchmark report: {filename}")
            return True
        return False

    def clear_reports(self) -> int:
        """Удаляет все отчеты."""
        reports_dir = settings.BASE_DIR / "data" / "benchmark" / "retrieval_reports"
        if not reports_dir.exists():
            return 0
            
        count = 0
        for file in reports_dir.glob("retrieval_report_*.json"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete retrieval report {file.name}: {e}")
        
        logger.info(f"Cleared {count} retrieval benchmark reports")
        return count
