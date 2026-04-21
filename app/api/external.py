from fastapi import APIRouter, HTTPException, Query, Response, File, UploadFile
from pydantic import BaseModel
import dicttoxml
import base64
import json
from typing import Dict, Any, List, Optional

from app.services.ocr_service import OCRService
from app.services.cleaner_service import CleanerService
from app.services.structuring_service import StructuringService
from app.services.vector_service import VectorService
from app.services.recognition_service import RecognitionService

router = APIRouter()

# Dependency injection for services
ocr_service = OCRService()
cleaner_service = CleanerService()
vector_service = VectorService()
structuring_service = StructuringService(vector_service)
recognition_service = RecognitionService(ocr_service, cleaner_service, structuring_service)


class RecognitionRequest(BaseModel):
    image_base64: str

class TaskStartedResponse(BaseModel):
    job_id: str
    status: str = "pending"

class TaskResultResponse(BaseModel):
    status: str
    value: Optional[Any] = None
    error: Optional[str] = None


class LearnResponse(BaseModel):
    status: str
    example_id: str


class LearnRequest(BaseModel):
    xml: str
    structured_data: Dict[str, Any]
    doc_type: Optional[str] = None


class ExampleSummary(BaseModel):
    id: str
    doc_type: Optional[str] = None
    text_preview: str
    created_at: Optional[str] = None


class ExampleDetail(BaseModel):
    id: str
    text: str
    json_output: str
    doc_type: Optional[str] = None
    created_at: Optional[str] = None


class ExampleUpdateRequest(BaseModel):
    json_output: Optional[str] = None
    doc_type: Optional[str] = None


@router.post("/recognize", response_model=TaskStartedResponse)
async def start_document_recognition(request: RecognitionRequest):
    """
    Отправляет изображение на полный цикл распознавания: OCR -> Cleaner -> RAG -> LLM.
    Возвращает job_id для отслеживания статуса.
    """
    if not request.image_base64:
         raise HTTPException(status_code=400, detail="image_base64 field is required")
         
    task_id = await recognition_service.start_recognition_task(request.image_base64)
    return {"job_id": task_id}


@router.post("/generate-ocr", response_model=TaskStartedResponse)
async def generate_ocr(file: UploadFile = File(..., description="Изображение документа для запуска OCR-пайплайна")):
    """
    Загрузка изображения документа для запуска OCR-пайплайна.
    """
    content = await file.read()
    if not content:
         raise HTTPException(status_code=400, detail="Empty file provided")
         
    image_base64 = base64.b64encode(content).decode('utf-8')
    task_id = await recognition_service.start_template_task(image_base64)
    return {"job_id": task_id}


@router.get("/result/{job_id}", response_model=TaskResultResponse)
async def get_job_result(job_id: str):
    """
    Получение результата асинхронной задачи по её идентификатору.
    Поддерживает как OCR (/generate-ocr), так и структурирование (/recognize).
    """
    result_data = await recognition_service.get_task_result(job_id)
    if not result_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    internal_status = result_data["status"]
    
    # Map status: pending/processing -> pending, completed -> done, error -> error
    if internal_status in ["pending", "processing"]:
        status = "pending"
    elif internal_status == "completed":
        status = "done"
    else:
        status = "error"
        
    value = None
    error = result_data.get("error")
    
    if status == "done":
        value = result_data.get("result")
            
    return {
        "status": status,
        "value": value,
        "error": error
    }


@router.post("/learn", response_model=LearnResponse)
async def learn_example(request: LearnRequest):
    """
    Добавляет новый пример (XML + структурированные данные) в базу знаний.
    """
    try:
        # Извлекаем текст из XML
        cleaned_text = cleaner_service.parse_xml_bytes(request.xml.encode("utf-8"))
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from XML")
            
        json_output = json.dumps(request.structured_data, ensure_ascii=False)
        
        # Добавляем в векторную базу
        example_id = await vector_service.add_example(
            cleaned_text, 
            json_output, 
            doc_type=request.doc_type
        )
        
        return {"status": "success", "example_id": example_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/examples", response_model=List[ExampleSummary])
async def list_examples():
    """
    Возвращает список всех примеров с краткой информацией.
    """
    return await vector_service.get_examples()


@router.get("/examples/{example_id}", response_model=ExampleDetail)
async def get_example(example_id: str):
    """
    Возвращает полный пример по ID.
    """
    example = await vector_service.get_example(example_id)
    if not example:
        raise HTTPException(status_code=404, detail="Example not found")
    return example


@router.put("/examples/{example_id}", response_model=ExampleDetail)
async def update_example(example_id: str, request: ExampleUpdateRequest):
    """
    Обновляет пример (json_output, doc_type). Автоматически переиндексирует в Qdrant.
    """
    updated = await vector_service.update_example(
        example_id,
        json_output=request.json_output,
        doc_type=request.doc_type
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Example not found")
    return updated


@router.delete("/examples/{example_id}")
async def delete_example(example_id: str):
    """
    Удаляет пример из PostgreSQL и Qdrant.
    """
    deleted = await vector_service.delete_example(example_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Example not found")
    return {"status": "deleted", "id": example_id}
