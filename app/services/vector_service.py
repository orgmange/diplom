import uuid
import hashlib
import ollama
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings

class VectorService:
    """
    Сервис для векторизации текста через Ollama и взаимодействия с Qdrant.
    """
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
        """ Генерирует стабильный UUID на основе ключа (например, имени файла). """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def ensure_collection(self):
        try:
            self.client.get_collection(settings.COLLECTION_NAME)
        except Exception:
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )

    def vectorize_text(self, text: str) -> List[float]:
        response = self.ollama_client.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']

    def index_examples(self) -> List[str]:
        """
        Индексирует примеры из data/examples.
        Ищет пары *_input.txt и *_output.json.
        """
        self.ensure_collection()
        examples_dir = settings.DATA_DIR / "examples"
        if not examples_dir.exists():
            return []

        indexed = []
        points = []
        
        # Находим все input файлы
        input_files = list(examples_dir.glob("*_input.txt"))
        for input_path in input_files:
            try:
                # Определяем базовое имя (например, passport_example)
                base_name = input_path.name.replace("_input.txt", "")
                output_path = examples_dir / f"{base_name}_output.json"
                
                if not output_path.exists():
                    continue

                with open(input_path, "r", encoding="utf-8") as f:
                    raw_text = f.read().strip()
                
                with open(output_path, "r", encoding="utf-8") as f:
                    json_output = f.read().strip()

                if not raw_text:
                    continue

                # Для примеров пока считаем cleaned_text = raw_text, 
                # если нет отдельного файла (или можно прогнать через cleaner)
                cleaned_text = raw_text 
                
                point_id = self.generate_id(f"example_{base_name}")
                vector = self.vectorize_text(raw_text)
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "filename": input_path.name,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "json_output": json_output,
                        "is_example": True
                    }
                ))
                indexed.append(input_path.name)
            except Exception as e:
                print(f"Error indexing example {input_path.name}: {e}")

        if points:
            self.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points
            )
        return indexed

    def index_directory(self) -> List[str]:
        """
        Сканирует директорию OCR и индексирует новые файлы.
        Использует *-xml как raw_text и *-clean как cleaned_text.
        """
        self.ensure_collection()
        
        indexed_files = []
        if not settings.OCR_DIR.exists():
            return []
            
        xml_files = list(settings.OCR_DIR.glob("*-xml"))
        
        points = []
        for xml_path in xml_files:
            try:
                clean_path = xml_path.parent / xml_path.name.replace("-xml", "-clean")
                if not clean_path.exists():
                    continue

                point_id = self.generate_id(xml_path.name)
                
                with open(xml_path, "r", encoding="utf-8") as f:
                    raw_text = f.read().strip()
                
                with open(clean_path, "r", encoding="utf-8") as f:
                    cleaned_text = f.read().strip()
                
                if not raw_text:
                    continue
                
                vector = self.vectorize_text(raw_text)
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "filename": xml_path.name,
                        "raw_text": raw_text,
                        "cleaned_text": cleaned_text,
                        "is_example": False
                    }
                ))
                indexed_files.append(xml_path.name)
            except Exception as e:
                print(f"Error indexing {xml_path.name}: {e}")

        if points:
            self.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points
            )
            
        return indexed_files

    def search(self, query: str, limit: int = 5, only_examples: bool = False) -> List[Dict]:
        self.ensure_collection()
        vector = self.vectorize_text(query)
        
        filter_query = None
        if only_examples:
            filter_query = models.Filter(
                must=[models.FieldCondition(key="is_example", match=models.MatchValue(value=True))]
            )

        results = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector,
            limit=limit,
            query_filter=filter_query
        ).points
        
        return [
            {
                "filename": res.payload.get("filename"),
                "raw_text": res.payload.get("raw_text"),
                "cleaned_text": res.payload.get("cleaned_text"),
                "json_output": res.payload.get("json_output"),
                "score": res.score
            }
            for res in results
        ]
