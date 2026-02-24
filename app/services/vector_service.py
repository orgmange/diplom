import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings


class VectorService:
    """Сервис для работы с векторами и коллекцией Qdrant."""

    def __init__(self):
        self._client = None
        self._ollama_client = None

    @property
    def client(self) -> QdrantClient:
        if not self._client:
            self._client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        return self._client

    @property
    def ollama_client(self):
        if not self._ollama_client:
            self._ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        return self._ollama_client

    def generate_id(self, key: str) -> str:
        """Генерирует стабильный UUID на основе ключа."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def ensure_collection(self, vector_size: int = 768):
        try:
            self.client.get_collection(settings.COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def reset_collection(self, vector_size: int = 768):
        """Очищает коллекцию и создает ее заново."""
        try:
            self.client.delete_collection(settings.COLLECTION_NAME)
        except Exception:
            pass
        self.ensure_collection(vector_size=vector_size)

    def list_embedding_models(self) -> List[str]:
        """Возвращает список доступных embedding-моделей Ollama."""
        payload = self.ollama_client.list()
        items = payload.get("models", []) if isinstance(payload, dict) else getattr(payload, "models", [])
        names: List[str] = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name")
                model = item.get("model")
                if isinstance(name, str):
                    names.append(name)
                elif isinstance(model, str):
                    names.append(model)
                continue
            name = getattr(item, "name", None)
            model = getattr(item, "model", None)
            if isinstance(name, str):
                names.append(name)
            elif isinstance(model, str):
                names.append(model)
        embedding_models = [name for name in names if "embed" in name.lower() or "embedding" in name.lower()]
        return sorted(set(embedding_models or names))

    def get_embedding_size(self, embedding_model: Optional[str] = None) -> int:
        """Возвращает размер вектора для embedding-модели."""
        vector = self.vectorize_text("test", embedding_model=embedding_model)
        return len(vector)

    def vectorize_text(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        """Преобразует текст в вектор с выбранной embedding-моделью."""
        truncated_text = text[:3000]
        response = self.ollama_client.embeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            prompt=truncated_text,
        )
        return response["embedding"]

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

    def _iter_doc_dirs(self) -> List[Path]:
        if not settings.DOCS_DIR.exists():
            return []
        return sorted(path for path in settings.DOCS_DIR.iterdir() if path.is_dir())

    def _upsert_points(self, points: List[models.PointStruct]):
        if not points:
            return
        self.client.upsert(collection_name=settings.COLLECTION_NAME, points=points)

    def index_examples(self, embedding_model: Optional[str] = None) -> List[str]:
        """Индексирует примеры из data/examples."""
        self.ensure_collection()
        examples_dir = settings.DATA_DIR / "examples"
        if not examples_dir.exists():
            return []

        indexed: List[str] = []
        points: List[models.PointStruct] = []
        for input_path in examples_dir.glob("*_input.txt"):
            base_name = input_path.name.replace("_input.txt", "")
            output_path = examples_dir / f"{base_name}_output.json"
            if not output_path.exists():
                continue
            raw_text = input_path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            json_output = output_path.read_text(encoding="utf-8").strip()
            vector = self.vectorize_text(raw_text, embedding_model=embedding_model)
            points.append(
                models.PointStruct(
                    id=self.generate_id(f"example_{base_name}"),
                    vector=vector,
                    payload={
                        "filename": input_path.name,
                        "raw_text": raw_text,
                        "cleaned_text": raw_text,
                        "json_output": json_output,
                        "is_example": True,
                        "is_cleaned": False,
                        "source_mode": "example",
                        "doc_type": self._detect_doc_type(input_path.name),
                    },
                )
            )
            indexed.append(input_path.name)
        self._upsert_points(points)
        return indexed

    def index_directory(self, embedding_model: Optional[str] = None) -> List[str]:
        """Индексирует пары *-xml и *-clean из data/ocr."""
        self.ensure_collection()
        indexed_files: List[str] = []
        if not settings.OCR_DIR.exists():
            return []

        points: List[models.PointStruct] = []
        for xml_path in settings.OCR_DIR.glob("*-xml"):
            clean_path = xml_path.parent / xml_path.name.replace("-xml", "-clean")
            if not clean_path.exists():
                continue
            raw_text = xml_path.read_text(encoding="utf-8").strip()
            cleaned_text = clean_path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            vector = self.vectorize_text(raw_text, embedding_model=embedding_model)
            points.append(
                models.PointStruct(
                    id=self.generate_id(xml_path.name),
                    vector=vector,
                    payload={
                        "filename": xml_path.name,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "is_example": False,
                        "is_cleaned": False,
                        "source_mode": "raw",
                        "doc_type": self._detect_doc_type(xml_path.name),
                    },
                )
            )
            indexed_files.append(xml_path.name)
        self._upsert_points(points)
        return indexed_files

    def index_templates_by_mode(
        self,
        is_cleaned: bool,
        directory: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ) -> List[str]:
        """Индексирует шаблоны из директории для выбранного типа текста."""
        self.ensure_collection()
        source_dir = directory or settings.OCR_DIR
        if not source_dir.exists():
            return []
        pattern = "*-clean" if is_cleaned else "*-xml"
        mode = "clean" if is_cleaned else "raw"
        indexed: List[str] = []
        points: List[models.PointStruct] = []
        for path in source_dir.glob(pattern):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            vector = self.vectorize_text(text, embedding_model=embedding_model)
            points.append(
                models.PointStruct(
                    id=self.generate_id(f"{mode}_{path.name}"),
                    vector=vector,
                    payload={
                        "filename": path.name,
                        "raw_text": text if not is_cleaned else "",
                        "cleaned_text": text if is_cleaned else "",
                        "is_example": False,
                        "is_cleaned": is_cleaned,
                        "source_mode": mode,
                        "doc_type": self._detect_doc_type(path.name),
                    },
                )
            )
            indexed.append(path.name)
        self._upsert_points(points)
        return indexed

    def index_docs_directory(self, embedding_model: Optional[str] = None) -> List[str]:
        """Индексирует данные из data/docs/*/xml и data/docs/*/clean."""
        self.ensure_collection()
        indexed: List[str] = []
        points: List[models.PointStruct] = []
        for doc_dir in self._iter_doc_dirs():
            doc_type = doc_dir.name
            xml_dir = doc_dir / "xml"
            clean_dir = doc_dir / "clean"
            if xml_dir.exists():
                for xml_path in sorted(path for path in xml_dir.iterdir() if path.is_file()):
                    raw_text = xml_path.read_text(encoding="utf-8").strip()
                    if not raw_text:
                        continue
                    vector = self.vectorize_text(raw_text, embedding_model=embedding_model)
                    points.append(
                        models.PointStruct(
                            id=self.generate_id(f"docs_raw_{doc_type}_{xml_path.name}"),
                            vector=vector,
                            payload={
                                "filename": f"{doc_type}/{xml_path.name}",
                                "raw_text": raw_text,
                                "cleaned_text": "",
                                "is_example": False,
                                "is_cleaned": False,
                                "source_mode": "raw",
                                "doc_type": doc_type,
                            },
                        )
                    )
                    indexed.append(f"{doc_type}/{xml_path.name}")
            if clean_dir.exists():
                for clean_path in sorted(path for path in clean_dir.iterdir() if path.is_file()):
                    clean_text = clean_path.read_text(encoding="utf-8").strip()
                    if not clean_text:
                        continue
                    vector = self.vectorize_text(clean_text, embedding_model=embedding_model)
                    points.append(
                        models.PointStruct(
                            id=self.generate_id(f"docs_clean_{doc_type}_{clean_path.name}"),
                            vector=vector,
                            payload={
                                "filename": f"{doc_type}/{clean_path.name}",
                                "raw_text": "",
                                "cleaned_text": clean_text,
                                "is_example": False,
                                "is_cleaned": True,
                                "source_mode": "clean",
                                "doc_type": doc_type,
                            },
                        )
                    )
                    indexed.append(f"{doc_type}/{clean_path.name}")
        self._upsert_points(points)
        return indexed

    def _merge_filters(
        self,
        only_examples: bool,
        query_filter: Optional[models.Filter],
    ) -> Optional[models.Filter]:
        must_conditions: List[Any] = []
        if query_filter and query_filter.must:
            must_conditions.extend(query_filter.must)
        if only_examples:
            must_conditions.append(
                models.FieldCondition(key="is_example", match=models.MatchValue(value=True))
            )
        if not must_conditions:
            return query_filter
        return models.Filter(must=must_conditions)

    def reindex_all(self, embedding_model: Optional[str] = None) -> Dict[str, Any]:
        """Полностью очищает и переиндексирует все данные (примеры, OCR и документы)."""
        vector_size = self.get_embedding_size(embedding_model=embedding_model)
        self.reset_collection(vector_size=vector_size)
        
        indexed_examples = self.index_examples(embedding_model=embedding_model)
        indexed_ocr = self.index_directory(embedding_model=embedding_model)
        indexed_docs = self.index_docs_directory(embedding_model=embedding_model)
        
        return {
            "embedding_model": embedding_model or settings.EMBEDDING_MODEL,
            "indexed_examples": len(indexed_examples),
            "indexed_ocr": len(indexed_ocr),
            "indexed_docs": len(indexed_docs),
            "total": len(indexed_examples) + len(indexed_ocr) + len(indexed_docs)
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        only_examples: bool = False,
        embedding_model: Optional[str] = None,
        query_filter: Optional[models.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Ищет похожие документы по смыслу запроса."""
        self.ensure_collection()
        vector = self.vectorize_text(query, embedding_model=embedding_model)
        merged_filter = self._merge_filters(only_examples=only_examples, query_filter=query_filter)
        results = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            limit=limit,
            query_filter=merged_filter,
        ).points
        return [
            {
                "filename": res.payload.get("filename"),
                "raw_text": res.payload.get("raw_text"),
                "cleaned_text": res.payload.get("cleaned_text"),
                "text": res.payload.get("text"),
                "json_output": res.payload.get("json_output"),
                "doc_type": res.payload.get("doc_type"),
                "is_cleaned": res.payload.get("is_cleaned"),
                "source_mode": res.payload.get("source_mode"),
                "score": res.score,
            }
            for res in results
        ]
