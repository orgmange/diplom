from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

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
    task_id: str
    status: str = "pending"


@router.post("/templates", response_model=TaskStartedResponse)
def start_template_recognition(request: RecognitionRequest):
    """
    Отправляет изображение на OCR для получения XML разметки (используется для создания шаблона).
    Возвращает task_id для отслеживания статуса.
    """
    if not request.image_base64:
         raise HTTPException(status_code=400, detail="image_base64 field is required")
         
    task_id = recognition_service.start_template_task(request.image_base64)
    return {"task_id": task_id}


@router.get("/templates/{task_id}/status")
def get_template_task_status(task_id: str):
    """
    Возвращает текущий статус задачи распознавания шаблона.
    """
    status = recognition_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/templates/{task_id}/result")
def get_template_task_result(task_id: str):
    """
    Возвращает результат распознавания шаблона (XML).
    Доступно только когда статус = completed.
    """
    result = recognition_service.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not completed. Current status: {result['status']}"
        )
        
    return result


@router.post("/recognize", response_model=TaskStartedResponse)
def start_document_recognition(request: RecognitionRequest):
    """
    Отправляет изображение на полный цикл распознавания: OCR -> Cleaner -> RAG -> LLM.
    Возвращает task_id для отслеживания статуса.
    """
    if not request.image_base64:
         raise HTTPException(status_code=400, detail="image_base64 field is required")
         
    task_id = recognition_service.start_recognition_task(request.image_base64)
    return {"task_id": task_id}


@router.get("/recognize/{task_id}/status")
def get_document_recognition_status(task_id: str):
    """
    Возвращает текущий статус полного цикла распознавания.
    """
    status = recognition_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/recognize/{task_id}/result")
def get_document_recognition_result(task_id: str):
    """
    Возвращает итоговый структурированный JSON результат распознавания документа.
    Доступно только когда статус = completed.
    """
    result = recognition_service.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if result["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not completed. Current status: {result['status']}"
        )
        
    return result
