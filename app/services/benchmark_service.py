from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client.http import models

from app.core.config import settings
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

    def run(self, embedding_model: str) -> Dict[str, Any]:
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
        indexed_files = self.vector_service.index_examples(
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
            
        return self._format_run_result(embedding_model, indexed, clean_report)

    def _format_run_result(self, embedding_model, indexed, clean_report):
        return {
            "embedding_model": embedding_model,
            "indexed": indexed,
            "overall": {
                "total": clean_report.get("total", 0),
                "correct": clean_report.get("correct", 0),
                "accuracy": clean_report.get("accuracy", 0.0)
            },
            "clean_tests": clean_report,
        }
