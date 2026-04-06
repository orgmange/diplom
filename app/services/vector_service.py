import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sqlalchemy import select

from app.core.config import settings
from app.db.database import async_session_local
from app.db.models import Example
from app.services.utils import detect_doc_type

logger = logging.getLogger("diplom")


class VectorService:
    """Сервис для работы с векторами и коллекцией Qdrant."""

    def __init__(self):
        self._client = None
        self._ollama_client = None
        self._ollama_embed_client = None
        self._state_path = settings.DATA_DIR / "rag" / "state.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_sizes: Dict[str, int] = {}

    def _save_state(self, model_name: str):
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump({"current_model": model_name}, f)

    def _load_state(self) -> str:
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("current_model", settings.EMBEDDING_MODEL)
            except Exception:
                pass
        return settings.EMBEDDING_MODEL

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

    @property
    def ollama_embed_client(self):
        if not self._ollama_embed_client:
            self._ollama_embed_client = ollama.Client(host=settings.OLLAMA_EMBED_BASE_URL)
        return self._ollama_embed_client

    def generate_id(self, key: str) -> str:
        """Генерирует стабильный UUID на основе ключа."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def ensure_collection(self, vector_size: Optional[int] = None, collection_name: Optional[str] = None, embedding_model: Optional[str] = None):
        target_size = vector_size or self.get_embedding_size(embedding_model=embedding_model)
        name = collection_name or settings.COLLECTION_NAME
        try:
            current_info = self.client.get_collection(name)
            current_size = current_info.config.params.vectors.size
            if current_size != target_size:
                logger.warning(f"Vector size mismatch in {name} (expected {target_size}, got {current_size}). Resetting collection.")
                self.reset_collection(vector_size=target_size, collection_name=name)
        except Exception:
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=target_size, distance=models.Distance.COSINE),
            )

    def reset_collection(self, vector_size: Optional[int] = None, collection_name: Optional[str] = None, embedding_model: Optional[str] = None):
        """Очищает коллекцию и создает ее заново."""
        name = collection_name or settings.COLLECTION_NAME
        target_size = vector_size or self.get_embedding_size(embedding_model=embedding_model)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.ensure_collection(vector_size=target_size, collection_name=name)

    def list_embedding_models(self) -> List[str]:
        """Возвращает список доступных embedding-моделей Ollama (с порта 11435)."""
        payload = self.ollama_embed_client.list()
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
        """Возвращает размер вектора для embedding-модели с кэшированием."""
        model = embedding_model or self._load_state()
        if model not in self._embedding_sizes:
            vector = self.vectorize_text("test", embedding_model=model)
            self._embedding_sizes[model] = len(vector)
        return self._embedding_sizes[model]

    def vectorize_text(self, text: str, embedding_model: Optional[str] = None) -> List[float]:
        """Преобразует текст в вектор с выбранной embedding-моделью (используя порт 11435)."""
        truncated_text = text[:3000]
        model = embedding_model or self._load_state()
        response = self.ollama_embed_client.embeddings(
            model=model,
            prompt=truncated_text,
        )
        return response["embedding"]



    def _iter_doc_dirs(self) -> List[Path]:
        if not settings.DOCS_DIR.exists():
            return []
        return sorted(path for path in settings.DOCS_DIR.iterdir() if path.is_dir())

    def _upsert_points(self, points: List[models.PointStruct], collection_name: Optional[str] = None):
        if not points:
            return
        self.client.upsert(collection_name=collection_name or settings.COLLECTION_NAME, points=points)

    async def migrate_examples_to_db(self):
        """Переносит примеры из data/examples в PostgreSQL, если их там ещё нет."""
        examples_dir = settings.DATA_DIR / "examples"
        if not examples_dir.exists():
            return

        async with async_session_local() as session:
            # Получаем список уже существующих текстов для простейшей проверки на дубликаты
            stmt = select(Example.text)
            result = await session.execute(stmt)
            existing_texts = set(result.scalars().all())

            added_count = 0
            for input_path in examples_dir.glob("*_input.txt"):
                raw_text = input_path.read_text(encoding="utf-8").strip()
                if not raw_text or raw_text in existing_texts:
                    continue

                base_name = input_path.name.replace("_input.txt", "")
                output_path = examples_dir / f"{base_name}_output.json"
                if not output_path.exists():
                    continue

                json_output = output_path.read_text(encoding="utf-8").strip()
                doc_type = detect_doc_type(input_path.name)

                new_example = Example(
                    id=str(uuid.uuid4()),
                    text=raw_text,
                    json_output=json_output,
                    doc_type=doc_type
                )
                session.add(new_example)
                added_count += 1
                existing_texts.add(raw_text)

            if added_count > 0:
                await session.commit()
                logger.info(f"Migrated {added_count} examples from disk to DB")

    async def index_examples(self, embedding_model: Optional[str] = None, collection_name: Optional[str] = None) -> List[str]:
        """Индексирует примеры из базы данных (предварительно мигрируя из файлов)."""
        # Сначала мигрируем новые файлы в базу
        await self.migrate_examples_to_db()

        self.ensure_collection(collection_name=collection_name, embedding_model=embedding_model)
        model_name = embedding_model or self._load_state()

        indexed_ids: List[str] = []
        async with async_session_local() as session:
            stmt = select(Example)
            result = await session.execute(stmt)
            db_examples = result.scalars().all()

            points = []
            for ex in db_examples:
                vector = self.vectorize_text(ex.text, embedding_model=model_name)
                points.append(
                    models.PointStruct(
                        id=self.generate_id(f"example_{ex.id}"),
                        vector=vector,
                        payload={
                            "text": ex.text,
                            "json_output": ex.json_output,
                            "doc_type": ex.doc_type,
                            "is_example": True,
                            "is_cleaned": True,
                        },
                    )
                )
                indexed_ids.append(ex.id)

            if points:
                self._upsert_points(points, collection_name=collection_name)

        return indexed_ids

    async def reindex_all(self, embedding_model: Optional[str] = None) -> Dict[str, Any]:
        """Полностью очищает и переиндексирует базу, используя только примеры из БД."""
        model_name = embedding_model or self._load_state()
        vector_size = self.get_embedding_size(embedding_model=model_name)
        self.reset_collection(vector_size=vector_size)
        
        # Индексируем примеры (включая миграцию из файлов)
        indexed_names = await self.index_examples(embedding_model=model_name)
        
        self._save_state(model_name)
        
        return {
            "embedding_model": model_name,
            "total_indexed": len(indexed_names),
            "indexed_names": indexed_names
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        embedding_model: Optional[str] = None,
        query_filter: Optional[models.Filter] = None,
        collection_name: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Ищет похожие документы в единой базе примеров."""
        name = collection_name or settings.COLLECTION_NAME
        self.ensure_collection(collection_name=name, embedding_model=embedding_model)
        vector = self.vectorize_text(query, embedding_model=embedding_model)
        
        # Если указан doc_type, добавляем его в фильтр
        if doc_type:
            if not query_filter:
                query_filter = models.Filter(must=[])
            query_filter.must.append(
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))
            )

        results = self.client.query_points(
            collection_name=name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        ).points
        
        return [
            {
                "filename": res.payload.get("filename"),
                "text": res.payload.get("text"),
                "json_output": res.payload.get("json_output"),
                "doc_type": res.payload.get("doc_type"),
                "score": res.score,
            }
            for res in results
        ]

    async def add_example(self, raw_text: str, json_output: str, doc_type: Optional[str] = None) -> str:
        """Добавляет новый пример в базу данных и индексирует его в Qdrant."""
        example_id = str(uuid.uuid4())
        
        # 1. Детектируем doc_type
        if not doc_type:
            try:
                data = json.loads(json_output)
                if isinstance(data, dict):
                    doc_type = data.get("doc_type")
            except:
                pass
            if not doc_type:
                doc_type = detect_doc_type(example_id)

        # 2. Сохраняем в PostgreSQL
        async with async_session_local() as session:
            new_example = Example(
                id=example_id,
                text=raw_text,
                json_output=json_output,
                doc_type=doc_type
            )
            session.add(new_example)
            await session.commit()
            
        # 3. Индексируем в Qdrant
        vector = self.vectorize_text(raw_text)
        point = models.PointStruct(
            id=self.generate_id(f"example_{example_id}"),
            vector=vector,
            payload={
                "text": raw_text,
                "json_output": json_output,
                "doc_type": doc_type,
                "is_example": True,
                "is_cleaned": True,
            },
        )
        
        self._upsert_points([point])
        return example_id
