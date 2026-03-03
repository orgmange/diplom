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
    doc_type: str | None = None

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


class StructuringBenchmarkMultiRunRequest(BaseModel):
    model_names: List[str]
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
    expected_type: str
    detected_type: str
    is_type_correct: bool
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
    total_count: int
    files: List[str]


# BenchmarkPreparedResponse removed as we only use clean examples now


class BenchmarkOverallResponse(BaseModel):
    total: int
    correct: int
    accuracy: float


class BenchmarkRunResponse(BaseModel):
    embedding_model: str
    indexed: BenchmarkIndexedResponse
    overall: BenchmarkOverallResponse
    clean_tests: BenchmarkGroupResponse


class StructuringBenchmarkRunResponse(BaseModel):
    model_name: str
    embedding_model: str
    total_files: int
    files_with_reference: int
    correct_templates_count: int
    template_accuracy: float
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
async def list_models():
    """Список доступных моделей Ollama."""
    models = await structuring_service.get_available_models()
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
async def run_structuring_benchmark(request: StructuringBenchmarkRunRequest):
    """Запускает бенчмарк структурирования для выбранной LLM модели."""
    report = await structuring_benchmark_service.run(
        model_name=request.model_name,
        embedding_model=request.embedding_model
    )
    return report.to_dict()

@router.post("/rag/benchmark/structuring/run-multi", response_model=List[StructuringBenchmarkRunResponse])
async def run_structuring_benchmark_multi(request: StructuringBenchmarkMultiRunRequest):
    """Поочерёдно запускает бенчмарк структурирования для списка LLM моделей."""
    reports = await structuring_benchmark_service.run_multi(
        model_names=request.model_names,
        embedding_model=request.embedding_model
    )
    return [report.to_dict() for report in reports]

@router.post("/rag/benchmark/cancel")
def cancel_benchmark():
    """Прерывает выполнение текущих бенчмарков."""
    benchmark_service.stop()
    structuring_benchmark_service.stop()
    return {"status": "Cancellation requested"}

@router.post("/rag/benchmark/structuring/skip")
def skip_structuring_model():
    """Пропускает текущую модель в бенчмарке структурирования."""
    structuring_benchmark_service.skip_model()
    return {"status": "Skip requested"}

@router.get("/rag/benchmark/structuring/progress")
def get_structuring_benchmark_progress():
    """Возвращает текущий прогресс бенчмарка структурирования."""
    return structuring_benchmark_service.get_progress()

@router.get("/rag/benchmark/structuring/reports", response_model=List[Dict[str, Any]])
def list_structuring_reports():
    """Возвращает список сохраненных отчетов структурирования."""
    return structuring_benchmark_service.list_reports()

@router.get("/rag/benchmark/structuring/reports/{filename}", response_model=Dict[str, Any])
def get_structuring_report(filename: str):
    """Возвращает детали конкретного отчета структурирования."""
    report = structuring_benchmark_service.get_report(filename)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@router.delete("/rag/benchmark/structuring/reports/{filename}")
def delete_structuring_report(filename: str):
    """Удаляет конкретный отчет структурирования."""
    success = structuring_benchmark_service.delete_report(filename)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"status": "deleted", "filename": filename}

@router.delete("/rag/benchmark/structuring/reports")
def clear_structuring_reports():
    """Удаляет всю историю отчетов структурирования."""
    count = structuring_benchmark_service.clear_reports()
    return {"status": "cleared", "deleted_count": count}

@router.post("/rag/index_examples", response_model=Dict[str, List[str]])
def index_examples():
    """Векторизует примеры из data/examples и сохраняет их в Qdrant."""
    indexed = vector_service.index_examples()
    return {"indexed_examples": indexed}

@router.post("/rag/index", response_model=Dict[str, Any])
def index_documents():
    """Векторизует примеры из data/examples (теперь это единый источник)."""
    indexed = vector_service.index_examples()
    return {"indexed_files": indexed, "count": len(indexed)}

@router.post("/rag/reindex", response_model=Dict[str, Any])
def reindex_database(request: Dict[str, str]):
    """Полная переиндексация базы с выбранной моделью."""
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")
    return vector_service.reindex_all(embedding_model=model_name)

@router.post("/rag/search", response_model=List[Dict[str, Any]])
def search_documents(query: SearchQuery):
    """Ищет документы в едином хранилище примеров."""
    results = vector_service.search(
        query.query,
        query.limit,
        embedding_model=query.model,
        doc_type=query.doc_type
    )
    return results

@router.post("/rag/structure", response_model=Dict[str, Any])
async def structure_document(request: StructureRequest):
    """Выполняет структурирование данных для выбранного файла."""
    # 1. Загружаем данные файла
    from app.core.config import settings
    
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
            
    # 3. Выполняем структурирование асинхронно
    # Благодаря AsyncClient в сервисе, если клиент закроет соединение,
    # запрос к Ollama будет прерван автоматически библиотекой httpx/ollama
    result = await structuring_service.structure(
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
