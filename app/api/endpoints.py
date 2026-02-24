from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.ocr_service import OCRService
from app.services.cleaner_service import CleanerService
from app.services.vector_service import VectorService
from app.services.structuring_service import StructuringService
from app.services.benchmark_service import BenchmarkService

router = APIRouter()
ocr_service = OCRService()
cleaner_service = CleanerService()
vector_service = VectorService()
structuring_service = StructuringService(vector_service)
benchmark_service = BenchmarkService(vector_service)

class SearchQuery(BaseModel):
    query: str
    limit: int = 5
    model: str | None = None

class StructureRequest(BaseModel):
    filename: str  # Имя файла из data/ocr
    model_name: str = "llama3:latest"
    use_raw_for_search: bool = True
    reindex: bool = False

class StatusResponse(BaseModel):
    ocr_files: int
    cleaned_files: int
    vectorized_count: int


class BenchmarkRunRequest(BaseModel):
    embedding_model: str


class BenchmarkItemResponse(BaseModel):
    filename: str
    expected_type: str | None
    predicted_type: str | None
    predicted_filename: str | None
    score: float | None
    is_correct: bool


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

@router.post("/rag/search", response_model=List[Dict[str, Any]])
def search_documents(query: SearchQuery):
    """Ищет документы по смыслу."""
    results = vector_service.search(query.query, query.limit, embedding_model=query.model)
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
        count_res = vector_service.client.count(collection_name=settings.COLLECTION_NAME)
        vec_count = count_res.count
    except:
        vec_count = 0

    return {
        "ocr_files": ocr_count,
        "cleaned_files": clean_count,
        "vectorized_count": vec_count
    }
