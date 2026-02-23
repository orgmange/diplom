import os
import time
import base64
import requests
from typing import List
from pathlib import Path

def load_env():
    """
    Загружает переменные окружения из файла .env, находящегося в корне проекта.
    """
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

load_env()

SOURCE_DIR = os.path.join("data", "ocr")
API_KEY = os.environ.get("OCR_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://ocrbot.ru/api/v1"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

class ProtonOCR:
    """
    Класс для взаимодействия с API сервиса OCR.
    """
    def __init__(self, api_key: str, base_url: str):
        """
        Инициализирует клиент API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-Api-Key": self.api_key}

    def _encode_image(self, path: Path) -> str:
        """
        Кодирует изображение в строку base64.
        """
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def create_task(self, image_paths: List[Path]) -> str:
        """
        Создает задачу распознавания для списка изображений.
        """
        images_b64 = [self._encode_image(p) for p in image_paths]
        payload = {"image": images_b64, "return_type": "xml"}
        
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

        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id returned: {data}")

        return task_id

    def wait_for_task(self, task_id: str) -> str:
        """
        Ожидает завершения задачи, опрашивая API.
        """
        for _ in range(60):
            resp = requests.get(
                f"{self.base_url}/tasks/{task_id}/status",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            status = resp.json().get("task_status", "unknown")
            print(f"Status: {status}")

            if status == "success" or status == "error":
                return status
            time.sleep(1)
        
        raise TimeoutError("Task timed out")

    def fetch_result(self, task_id: str) -> List[bytes]:
        """
        Получает результат распознавания и декодирует его из base64.
        """
        resp = requests.get(
            f"{self.base_url}/tasks/{task_id}/result",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        
        raw_results = data.get("recognition_result", [])
        if isinstance(raw_results, str):
            raw_results = [raw_results]
            
        decoded_results = []
        for res in raw_results:
            try:
                decoded = base64.b64decode(res)
                decoded_results.append(decoded)
            except Exception as e:
                print(f"Error decoding result: {e}")
                
        return decoded_results

def get_candidates(base_path: Path) -> List[Path]:
    """
    Ищет изображения, для которых еще не созданы XML-файлы с результатом.
    """
    image_files = [
        f for f in base_path.iterdir() 
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    ]
    
    candidates = []
    for img in image_files:
        if not (img.parent / f"{img.name}-xml").exists():
            candidates.append(img)
    return candidates

def process_image(ocr: ProtonOCR, img_path: Path):
    """
    Выполняет полный цикл обработки одного изображения.
    """
    try:
        task_id = ocr.create_task([img_path])
        status = ocr.wait_for_task(task_id)
        
        if status != "success":
            print(f"Task failed for {img_path.name}")
            return
            
        results = ocr.fetch_result(task_id)
        if results:
            xml_out_path = img_path.parent / f"{img_path.name}-xml"
            with open(xml_out_path, "wb") as f:
                f.write(results[0])
            print(f"Saved XML: {xml_out_path.name}")
            
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

def main():
    """
    Основная точка входа: находит файлы и запускает процесс распознавания.
    """
    base_path = Path.cwd() / SOURCE_DIR
    if not base_path.exists():
        print(f"Error: Directory '{SOURCE_DIR}' not found.")
        return

    candidates = get_candidates(base_path)
    if not candidates:
        print("No new images to process.")
        return

    ocr = ProtonOCR(API_KEY, BASE_URL)
    for img_path in candidates:
        print(f"\n--- Processing {img_path.name} ---")
        process_image(ocr, img_path)

if __name__ == "__main__":
    main()
