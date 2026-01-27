import os
import time
import base64
import requests
from typing import List
from pathlib import Path

# --- Configuration ---
def load_env():
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
USE_MOCK = False  # Set to False to use real API

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

class ProtonOCR:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-Api-Key": self.api_key}

    def _encode_image(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def create_task(self, image_paths: List[Path]) -> str:
        """Creates an OCR task for the given images."""
        if USE_MOCK:
            print(f"[MOCK] Uploading {len(image_paths)} images...")
            time.sleep(1)
            return "mock-task-id-12345"

        images_b64 = [self._encode_image(p) for p in image_paths]
        
        payload = {
            "image": images_b64, 
            "return_type": "xml"
        }
        
        try:
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
        except Exception as e:
            print(f"Error creating task: {e}")
            raise

    def wait_for_task(self, task_id: str) -> str:
        """Waits for the task to complete."""
        if USE_MOCK:
            print(f"[MOCK] Waiting for task {task_id}...")
            time.sleep(1)
            print("[MOCK] status: success")
            return "success"

        for _ in range(60):  # Wait up to 60 seconds
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
        """Fetches and decodes the result."""
        if USE_MOCK:
            print(f"[MOCK] Fetching results for {task_id}")
            return [b"<xml>Mock XML</xml>"]

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

def main():
    base_path = Path.cwd() / SOURCE_DIR
    if not base_path.exists():
        print(f"Error: Directory '{SOURCE_DIR}' not found.")
        return

    # 1. Find images
    all_files = list(base_path.iterdir())
    image_files = [
        f for f in all_files 
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()
    ]

    # 2. Filter images that already have a -xml file
    candidates = []
    for img in image_files:
        expected_xml_name = f"{img.name}-xml"
        expected_xml_path = img.parent / expected_xml_name
        
        if not expected_xml_path.exists():
            candidates.append(img)
    
    if not candidates:
        print("No new images to process. All images already have corresponding -xml files.")
        return

    # 3. User Selection
    print(f"\nFound {len(candidates)} new images:")
    for i, img in enumerate(candidates):
        print(f"{i + 1}. {img.name}")
    
    # Automated selection for agent
    choice = 'all'
    print(f"\nAutomated selection: {choice}")
    
    selected_images = []
    if choice == 'all':
        selected_images = candidates
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(candidates):
                    selected_images.append(candidates[idx])
        except ValueError:
            print("Invalid input.")
            return

    if not selected_images:
        print("No images selected.")
        return

    # 4. Process Images
    ocr = ProtonOCR(API_KEY, BASE_URL)
    
    print(f"\nProcessing {len(selected_images)} images...")
    
    for img_path in selected_images:
        print(f"\n--- Processing {img_path.name} ---")
        try:
            task_id = ocr.create_task([img_path])
            
            status = ocr.wait_for_task(task_id)
            if status != "success":
                print(f"Task failed for {img_path.name}")
                continue
                
            results = ocr.fetch_result(task_id)
            if not results:
                print("No results returned.")
                continue
            
            xml_data = results[0]
            
            xml_filename = f"{img_path.name}-xml"
            xml_out_path = img_path.parent / xml_filename
            with open(xml_out_path, "wb") as f:
                f.write(xml_data)
            print(f"Saved XML: {xml_filename}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print("\nDone. Run 'ocr_cleaner.py' to parse the XML files.")

if __name__ == "__main__":
    main()