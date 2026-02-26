from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client.http import models
from app.services.ocr_service import OCRService
from app.services.cleaner_service import CleanerService
from app.services.vector_service import VectorService
from app.services.structuring_service import StructuringService
from app.services.benchmark_service import BenchmarkService
from app.services.structuring_benchmark_service import StructuringBenchmarkService

router = APIRouter()
ocr_service = OCRService()
cleaner_service = CleanerService()
vector_service = VectorService()
structuring_service = StructuringService(vector_service)
benchmark_service = BenchmarkService(vector_service)
structuring_benchmark_service = StructuringBenchmarkService(structuring_service)

class SearchQuery(BaseModel):
    query: str
    limit: int = 5
    model: str | None = None
    is_cleaned: bool | None = None
    only_templates: bool = False

class StructureRequest(BaseModel):
    filename: str  # Имя файла из data/ocr
    model_name: str = "llama3:latest"
    use_raw_for_search: bool = True
    reindex: bool = False

class StatusResponse(BaseModel):
    ocr_files: int
    cleaned_files: int
    vectorized_count: int
    current_model: str


class BenchmarkRunRequest(BaseModel):
    embedding_model: str


class StructuringBenchmarkRunRequest(BaseModel):
    model_name: str
    embedding_model: str | None = None


class BenchmarkItemResponse(BaseModel):
    filename: str
    expected_type: str | None
    predicted_type: str | None
    predicted_filename: str | None
    score: float | None
    is_correct: bool
    alternatives: List[Dict[str, Any]] | None = None


class StructuringBenchmarkItemResponse(BaseModel):
    filename: str
    doc_type: str
    processing_time: float
    accuracy: float
    is_reference_found: bool
    result_json: Dict[str, Any]
    reference_json: Dict[str, Any] | None


class BenchmarkGroupResponse(BaseModel):
    total: int
    correct: int
    accuracy: float
    items: List[BenchmarkItemResponse]


class BenchmarkIndexedResponse(BaseModel):
    raw_count: int
    clean_count: int
    total_count: int
    raw_files: List[str]
    clean_files: List[str]


class BenchmarkPreparedResponse(BaseModel):
    prepared_xml: int
    prepared_clean: int


class BenchmarkOverallResponse(BaseModel):
    total: int
    correct: int
    accuracy: float


class BenchmarkRunResponse(BaseModel):
    embedding_model: str
    prepared: BenchmarkPreparedResponse
    indexed: BenchmarkIndexedResponse
    overall: BenchmarkOverallResponse
    raw_tests: BenchmarkGroupResponse
    clean_tests: BenchmarkGroupResponse


class StructuringBenchmarkRunResponse(BaseModel):
    model_name: str
    embedding_model: str
    total_files: int
    files_with_reference: int
    avg_processing_time: float
    avg_accuracy: float
    items: List[StructuringBenchmarkItemResponse]


@router.post("/ocr/scan", response_model=Dict[str, List[str]])
def scan_ocr():
    """Сканирует data/docs/*/image и запускает OCR для новых изображений."""
    processed = ocr_service.process_docs_directory()
    return {"processed_files": processed}


@router.post("/ocr/clean", response_model=Dict[str, List[Dict[str, str]]])
def clean_ocr():
    """Очищает XML-файлы из data/docs/*/xml в data/docs/*/clean."""
    cleaned = cleaner_service.process_docs_directory()
    return {"cleaned_files": cleaned}


@router.get("/rag/models", response_model=Dict[str, List[str]])
def list_models():
    """Список доступных моделей Ollama."""
    models = structuring_service.get_available_models()
    return {"models": models}


@router.get("/rag/benchmark/models", response_model=Dict[str, List[str]])
def list_benchmark_models():
    """Список embedding-моделей для тестирования."""
    models = vector_service.list_embedding_models()
    return {"models": models}


@router.post("/rag/benchmark/run", response_model=BenchmarkRunResponse)
def run_benchmark(request: BenchmarkRunRequest):
    """Запускает полный цикл тестирования retrieval для embedding-модели."""
    return benchmark_service.run(request.embedding_model)

@router.post("/rag/benchmark/structuring/run", response_model=StructuringBenchmarkRunResponse)
def run_structuring_benchmark(request: StructuringBenchmarkRunRequest):
    """Запускает бенчмарк структурирования для выбранной LLM модели."""
    report = structuring_benchmark_service.run(
        model_name=request.model_name,
        embedding_model=request.embedding_model
    )
    return report.to_dict()

@router.post("/rag/index_examples", response_model=Dict[str, List[str]])
def index_examples():
    """Векторизует примеры из data/examples и сохраняет их в Qdrant."""
    indexed = vector_service.index_examples()
    return {"indexed_examples": indexed}

@router.post("/rag/index", response_model=Dict[str, List[str]])
def index_documents():
    """Векторизует очищенные тексты и сохраняет их в Qdrant."""
    indexed = vector_service.index_directory()
    return {"indexed_files": indexed}

@router.post("/rag/reindex", response_model=Dict[str, Any])
def reindex_database(request: Dict[str, str]):
    """Полная переиндексация базы с выбранной моделью."""
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    return vector_service.reindex_all(embedding_model=model_name)

@router.post("/rag/search", response_model=List[Dict[str, Any]])
def search_documents(query: SearchQuery):
    """Ищет документы по смыслу."""
    query_filter = None
    must_conditions = []
    
    if query.is_cleaned is not None:
        must_conditions.append(
            models.FieldCondition(key="is_cleaned", match=models.MatchValue(value=query.is_cleaned))
        )
    
    if query.only_templates:
        must_conditions.append(
            models.FieldCondition(key="is_example", match=models.MatchValue(value=False))
        )
        
    if must_conditions:
        query_filter = models.Filter(must=must_conditions)

    results = vector_service.search(
        query.query, 
        query.limit, 
        embedding_model=query.model,
        query_filter=query_filter
    )
    return results

@router.post("/rag/structure", response_model=Dict[str, Any])
def structure_document(request: StructureRequest):
    """Выполняет структурирование данных для выбранного файла."""
    # 1. Загружаем данные файла
    from app.core.config import settings
    import os
    
    xml_filename = request.filename.replace("-clean", "-xml")
    clean_filename = request.filename if "-clean" in request.filename else f"{request.filename}-clean"
    
    xml_path = settings.OCR_DIR / xml_filename
    clean_path = settings.OCR_DIR / clean_filename
    
    if not clean_path.exists():
        raise HTTPException(status_code=404, detail=f"Cleaned file {clean_filename} not found.")

    with open(clean_path, "r", encoding="utf-8") as f:
        cleaned_text = f.read().strip()
        
    raw_text = cleaned_text
    if xml_path.exists():
        with open(xml_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
            
    # 2. Переиндексация при необходимости
    if request.reindex:
        vector_service.reindex_all(embedding_model=request.model_name)
            
    # 3. Выполняем структурирование
    result = structuring_service.structure(
        raw_text=raw_text if request.use_raw_for_search else cleaned_text,
        cleaned_text=cleaned_text,
        model_name=request.model_name,
        embedding_model=request.model_name
    )
    
    return result

@router.get("/status", response_model=StatusResponse)
def get_status():
    """Возвращает текущую статистику."""
    from app.core.config import settings

    ocr_count = 0
    clean_count = 0
    if settings.DOCS_DIR.exists():
        for doc_dir in settings.DOCS_DIR.iterdir():
            if not doc_dir.is_dir():
                continue
            xml_dir = doc_dir / "xml"
            clean_dir = doc_dir / "clean"
            if xml_dir.exists():
                ocr_count += len([path for path in xml_dir.iterdir() if path.is_file()])
            if clean_dir.exists():
                clean_count += len([path for path in clean_dir.iterdir() if path.is_file()])
    elif settings.OCR_DIR.exists():
        ocr_count = len(list(settings.OCR_DIR.glob("*-xml")))
        clean_count = len(list(settings.OCR_DIR.glob("*-clean")))

    try:
        count_res = vector_service.client.count(collection_name=settings.COLLECTION_NAME, exact=True)
        vec_count = count_res.count
    except:
        vec_count = 0

    return {
        "ocr_files": ocr_count,
        "cleaned_files": clean_count,
        "vectorized_count": vec_count,
        "current_model": vector_service._load_state()
    }
