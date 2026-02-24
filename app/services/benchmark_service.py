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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "expected_type": self.expected_type,
            "predicted_type": self.predicted_type,
            "predicted_filename": self.predicted_filename,
            "score": self.score,
            "is_correct": self.is_correct,
        }


class BenchmarkService:
    """Сервис для полного теста retrieval по разным embedding-моделям."""

    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service

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

    def _evaluate(
        self,
        test_dir: Path,
        pattern: str,
        embedding_model: str,
        is_cleaned: bool,
    ) -> Dict[str, Any]:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="is_cleaned", match=models.MatchValue(value=is_cleaned)),
                models.FieldCondition(key="is_example", match=models.MatchValue(value=False)),
            ]
        )
        items: List[BenchmarkItem] = []
        for path in sorted(test_dir.glob(pattern)):
            query = path.read_text(encoding="utf-8").strip()
            expected_type = self._detect_doc_type(path.name)
            if not query:
                items.append(
                    BenchmarkItem(
                        filename=path.name,
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
                limit=1,
                embedding_model=embedding_model,
                query_filter=query_filter,
            )
            if not results:
                items.append(
                    BenchmarkItem(
                        filename=path.name,
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
            items.append(
                BenchmarkItem(
                    filename=path.name,
                    expected_type=expected_type,
                    predicted_type=predicted_type,
                    predicted_filename=predicted_filename,
                    score=score,
                    is_correct=expected_type == predicted_type if expected_type else False,
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
        """Запускает цикл reset, index и проверку raw/clean тестов."""
        template_dir = settings.OCR_DIR
        test_dir = settings.DATA_DIR / "ocr_backup"
        vector_size = self.vector_service.get_embedding_size(embedding_model=embedding_model)
        self.vector_service.reset_collection(vector_size=vector_size)
        indexed_raw = self.vector_service.index_templates_by_mode(
            is_cleaned=False,
            directory=template_dir,
            embedding_model=embedding_model,
        )
        indexed_clean = self.vector_service.index_templates_by_mode(
            is_cleaned=True,
            directory=template_dir,
            embedding_model=embedding_model,
        )
        raw_report = self._evaluate(
            test_dir=test_dir,
            pattern="*-xml",
            embedding_model=embedding_model,
            is_cleaned=False,
        )
        clean_report = self._evaluate(
            test_dir=test_dir,
            pattern="*-clean",
            embedding_model=embedding_model,
            is_cleaned=True,
        )
        return {
            "embedding_model": embedding_model,
            "indexed": {
                "raw_count": len(indexed_raw),
                "clean_count": len(indexed_clean),
                "total_count": len(indexed_raw) + len(indexed_clean),
                "raw_files": indexed_raw,
                "clean_files": indexed_clean,
            },
            "raw_tests": raw_report,
            "clean_tests": clean_report,
        }
