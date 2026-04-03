import os
import uuid
import base64
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from sqlalchemy import select, update

from app.core.config import settings
from app.services.ocr_service import OCRService
from app.services.cleaner_service import CleanerService
from app.services.structuring_service import StructuringService
from app.db.database import async_session_local
from app.db.models import Task

logger = logging.getLogger("diplom")

class RecognitionService:
    def __init__(
        self,
        ocr_service: OCRService,
        cleaner_service: CleanerService,
        structuring_service: StructuringService
    ):
        self.ocr_service = ocr_service
        self.cleaner_service = cleaner_service
        self.structuring_service = structuring_service
        
        # Ensure temp directory exists
        self.temp_dir = settings.DATA_DIR / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _save_base64_image(self, base64_str: str, task_id: str) -> Path:
        """Saves a base64 string to a temporary image file."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        image_data = base64.b64decode(base64_str)
        filepath = self.temp_dir / f"{task_id}.jpg"
        
        with open(filepath, "wb") as f:
            f.write(image_data)
            
        return filepath

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with async_session_local() as session:
            stmt = select(Task).where(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            if not task:
                return None
            return {"status": task.status}

    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with async_session_local() as session:
            stmt = select(Task).where(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                return None
            
            if task.status == "error":
                return {
                    "status": "error",
                    "error": task.error or "Unknown error occurred"
                }
            
            if task.status != "completed":
                return {"status": task.status}
                
            return {
                "status": "completed",
                "result": task.result
            }

    async def _update_task(self, task_id: str, status: str, result: Any = None, error: Optional[str] = None):
        async with async_session_local() as session:
            stmt = update(Task).where(Task.id == task_id).values(
                status=status,
                result=result,
                error=error
            )
            await session.execute(stmt)
            await session.commit()

    async def _process_template_task(self, task_id: str, image_path: Path):
        try:
            await self._update_task(task_id, "processing")
            
            # 1. OCR
            logger.info(f"Task {task_id}: Starting OCR")
            ocr_task_id = self.ocr_service.create_task(image_path)
            
            status = await asyncio.to_thread(self.ocr_service.wait_for_task, ocr_task_id)
            
            if status == "success":
                xml_bytes = await asyncio.to_thread(self.ocr_service.fetch_result, ocr_task_id)
                if xml_bytes:
                    xml_str = xml_bytes.decode("utf-8")
                    await self._update_task(task_id, "completed", result=xml_str)
                else:
                    raise ValueError("OCR returned empty result")
            else:
                raise ValueError("OCR task failed")
                
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self._update_task(task_id, "error", error=str(e))
        finally:
            if image_path.exists():
                image_path.unlink()

    async def _process_recognize_task(self, task_id: str, image_path: Path):
        try:
            await self._update_task(task_id, "processing")
            
            # 1. OCR
            logger.info(f"Task {task_id}: Starting OCR")
            ocr_task_id = await asyncio.to_thread(self.ocr_service.create_task, image_path)
            status = await asyncio.to_thread(self.ocr_service.wait_for_task, ocr_task_id)
            
            if status != "success":
                raise ValueError("OCR task failed")
                
            xml_bytes = await asyncio.to_thread(self.ocr_service.fetch_result, ocr_task_id)
            if not xml_bytes:
                raise ValueError("OCR returned empty result")
                
            # 2. Cleaner (Parse XML)
            logger.info(f"Task {task_id}: Extracting text from XML")
            cleaned_text = self.cleaner_service.parse_xml_bytes(xml_bytes)
            xml_text = xml_bytes.decode('utf-8')
            
            # 3. Structuring
            logger.info(f"Task {task_id}: Standardizing via LLM")
            model_name = settings.DEFAULT_MODEL
            
            result = await self.structuring_service.structure(
                raw_text=xml_text,
                cleaned_text=cleaned_text,
                model_name=model_name,
                structured_output=True
            )
            
            await self._update_task(task_id, "completed", result=result.get("result", {}))
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self._update_task(task_id, "error", error=str(e))
        finally:
            if image_path.exists():
                image_path.unlink()

    async def start_template_task(self, base64_image: str) -> str:
        task_id = str(uuid.uuid4())
        
        async with async_session_local() as session:
            new_task = Task(id=task_id, status="pending")
            session.add(new_task)
            await session.commit()
        
        try:
            image_path = self._save_base64_image(base64_image, task_id)
            asyncio.create_task(self._process_template_task(task_id, image_path))
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start task: {e}")
            await self._update_task(task_id, "error", error=str(e))
            return task_id

    async def start_recognition_task(self, base64_image: str) -> str:
        task_id = str(uuid.uuid4())
        
        async with async_session_local() as session:
            new_task = Task(id=task_id, status="pending")
            session.add(new_task)
            await session.commit()
        
        try:
            image_path = self._save_base64_image(base64_image, task_id)
            asyncio.create_task(self._process_recognize_task(task_id, image_path))
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start task: {e}")
            await self._update_task(task_id, "error", error=str(e))
            return task_id
