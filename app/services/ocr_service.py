import base64
import time
import requests
from pathlib import Path
from typing import List, Optional, Dict
from app.core.config import settings

class OCRService:
    """
    Сервис для взаимодействия с API OCR.
    """
    def __init__(self):
        self.api_key = settings.OCR_API_KEY
        self.base_url = settings.OCR_BASE_URL
        self.headers = {"X-Api-Key": self.api_key}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def _encode_image(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def create_task(self, image_path: Path) -> str:
        """Создает задачу распознавания."""
        if not image_path.exists():
            raise FileNotFoundError(f"File not found: {image_path}")

        payload = {
            "image": [self._encode_image(image_path)],
            "return_type": "xml"
        }
        
        resp = requests.post(
            f"{self.base_url}/tasks",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(f"API Error: {data}")

        return data.get("task_id")

    def wait_for_task(self, task_id: str, timeout: int = 60) -> str:
        """Ожидает завершения задачи."""
        for _ in range(timeout):
            resp = requests.get(
                f"{self.base_url}/tasks/{task_id}/status",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            status = resp.json().get("task_status", "unknown")
            
            if status in ["success", "error"]:
                return status
            time.sleep(1)
        
        raise TimeoutError("Task timed out")

    def fetch_result(self, task_id: str) -> Optional[bytes]:
        """Скачивает результат."""
        resp = requests.get(
            f"{self.base_url}/tasks/{task_id}/result",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        results = data.get("recognition_result", [])
        if not results:
            return None
            
        raw_b64 = results[0] if isinstance(results, list) else results
        return base64.b64decode(raw_b64)

    def process_directory(self) -> List[str]:
        """
        Сканирует директорию на наличие новых изображений и обрабатывает их.
        Возвращает список обработанных файлов.
        """
        processed_files = []
        if not settings.OCR_DIR.exists():
            return processed_files

        images = [
            f for f in settings.OCR_DIR.iterdir()
            if f.suffix.lower() in self.image_extensions and f.is_file()
        ]

        for img in images:
            xml_path = img.parent / f"{img.name}-xml"
            if xml_path.exists():
                continue

            try:
                task_id = self.create_task(img)
                status = self.wait_for_task(task_id)
                
                if status == "success":
                    result = self.fetch_result(task_id)
                    if result:
                        with open(xml_path, "wb") as f:
                            f.write(result)
                        processed_files.append(img.name)
            except Exception as e:
                print(f"Error processing {img.name}: {e}")

        return processed_files
