import os
import uuid
import base64
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.config import settings
from app.services.ocr_service import OCRService
from app.services.cleaner_service import CleanerService
from app.services.structuring_service import StructuringService

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
        
        # Simple in-memory task store for demonstration.
        # In a real production system, this should be Redis or a database.
        self._tasks: Dict[str, Dict[str, Any]] = {}
        
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

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        if task_id not in self._tasks:
            return None
        return {"status": self._tasks[task_id]["status"]}

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        if task_id not in self._tasks:
            return None
        task = self._tasks[task_id]
        if task["status"] != "completed":
            return {"status": task["status"]}
            
        # For template, 'result' holds XML string
        # For recognize, 'result' holds JSON dict
        return {
            "status": "completed",
            "result": task.get("result")
        }

    async def _process_template_task(self, task_id: str, image_path: Path):
        try:
            self._tasks[task_id]["status"] = "processing"
            
            # 1. OCR
            logger.info(f"Task {task_id}: Starting OCR")
            ocr_task_id = self.ocr_service.create_task(image_path)
            
            # Wait for OCR (sync call in an async function - ideally this should be async or run in threadpool)
            # using asyncio.to_thread to avoid blocking event loop
            status = await asyncio.to_thread(self.ocr_service.wait_for_task, ocr_task_id)
            
            if status == "success":
                xml_bytes = await asyncio.to_thread(self.ocr_service.fetch_result, ocr_task_id)
                if xml_bytes:
                    xml_str = xml_bytes.decode("utf-8")
                    self._tasks[task_id]["status"] = "completed"
                    self._tasks[task_id]["result"] = xml_str
                else:
                    raise ValueError("OCR returned empty result")
            else:
                raise ValueError("OCR task failed")
                
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self._tasks[task_id]["status"] = "error"
            self._tasks[task_id]["error"] = str(e)
        finally:
            if image_path.exists():
                image_path.unlink()

    async def _process_recognize_task(self, task_id: str, image_path: Path):
        try:
            self._tasks[task_id]["status"] = "processing"
            
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
            # Cleaner service just takes bytes and returns string, fast enough to be sync
            cleaned_text = self.cleaner_service.parse_xml_bytes(xml_bytes)
            xml_text = xml_bytes.decode('utf-8')
            
            # 3. Structuring
            logger.info(f"Task {task_id}: Standardizing via LLM")
            # Defaults. We can accept these as params later if needed.
            model_name = settings.DEFAULT_MODEL
            
            result = await self.structuring_service.structure(
                raw_text=xml_text,
                cleaned_text=cleaned_text,
                model_name=model_name,
                structured_output=True
            )
            
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["result"] = result.get("result", {})
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self._tasks[task_id]["status"] = "error"
            self._tasks[task_id]["error"] = str(e)
        finally:
            if image_path.exists():
                image_path.unlink()

    def start_template_task(self, base64_image: str) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {"status": "pending"}
        
        try:
            image_path = self._save_base64_image(base64_image, task_id)
            
            # Fire and forget async task
            asyncio.create_task(self._process_template_task(task_id, image_path))
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start task: {e}")
            self._tasks[task_id] = {"status": "error", "error": str(e)}
            return task_id

    def start_recognition_task(self, base64_image: str) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {"status": "pending"}
        
        try:
            image_path = self._save_base64_image(base64_image, task_id)
            
            # Fire and forget async task
            asyncio.create_task(self._process_recognize_task(task_id, image_path))
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start task: {e}")
            self._tasks[task_id] = {"status": "error", "error": str(e)}
            return task_id
