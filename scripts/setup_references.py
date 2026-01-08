import json
import os
from pathlib import Path
import shutil

# Пути относительно корня проекта
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "data/images"
OCR_DIR = BASE_DIR / "data/ocr"
REFS_DIR = BASE_DIR / "data/references"
TEMPLATES_DIR = BASE_DIR / "templates"

MAPPING = {
    "паспорт": "passport_ru.json",
    "права": "driver_license_ru.json",
    "снилс": "snils.json",
    "свидетельство": "birth_certificate_ru.json"
}

def setup_references():
    if not IMAGES_DIR.exists():
        print(f"Error: Directory {IMAGES_DIR} not found")
        return

    images = [f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    for img_path in images:
        template_name = None
        for keyword, t_name in MAPPING.items():
            if keyword in img_path.name.lower():
                template_name = t_name
                break
        
        if not template_name:
            continue

        template_path = TEMPLATES_DIR / template_name
        ref_path = REFS_DIR / f"{img_path.name}-reference.json"

        if not ref_path.exists():
            shutil.copy(template_path, ref_path)
            print(f"[+] Created reference template: data/references/{ref_path.name}")

    print("\nNext step: Open files in data/references/ and fill them with correct data.")

if __name__ == "__main__":
    setup_references()
