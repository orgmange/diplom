from ocr_processor import get_clean_text_from_xml
from pathlib import Path

path = Path("документы для тестирвания моделей/паспорт мой.png-xml")
clean_path = Path("документы для тестирвания моделей/паспорт мой.png-clean")

if path.exists():
    with open(path, "rb") as f:
        xml_data = f.read()
    
    clean_text = get_clean_text_from_xml(xml_data)
    
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
    
    print("Reprocessed clean file with spatial sorting.")
    print("--- NEW CONTENT ---")
    print(clean_text)
else:
    print("XML file not found.")
