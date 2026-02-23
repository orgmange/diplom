import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

SOURCE_DIR = os.path.join("data", "ocr")
MIN_CONFIDENCE = 0

def extract_raw_lines_from_xml(root: ET.Element, min_confidence: int) -> List[Dict]:
    """
    Извлекает данные о строках текста из XML-структуры, фильтруя их по уверенности распознавания.
    """
    lines_data = []
    text_lines = [elem for elem in root.iter() if elem.get("type") == "RIL_TEXTLINE"]

    for line_node in text_lines:
        try:
            cnf = int(line_node.get("cnf", 100))
        except ValueError:
            cnf = 100

        if cnf < min_confidence:
            continue

        try:
            y = int(line_node.get("Y", 0))
            x = int(line_node.get("X", 0))
            h = int(line_node.get("H", 10))
        except ValueError:
            y, x, h = 0, 0, 10

        words = [
            w.text.strip() for w in line_node.iter() 
            if w.get("type") == "RIL_WORD" and w.text and w.text.strip()
        ]

        if words:
            lines_data.append({"y": y, "x": x, "h": h, "text": " ".join(words)})
            
    return lines_data

def sort_and_group_lines(lines_data: List[Dict], threshold: int = 25) -> str:
    """
    Сортирует строки сначала по вертикали (Y), группирует их в одну строку при близости Y, 
    а затем сортирует каждую группу по горизонтали (X).
    """
    if not lines_data:
        return ""

    lines_data.sort(key=lambda k: k["y"])

    sorted_lines = []
    current_row = []
    current_row_y = -1000

    for line in lines_data:
        if abs(line["y"] - current_row_y) < threshold:
            current_row.append(line)
        else:
            if current_row:
                current_row.sort(key=lambda k: k["x"])
                sorted_lines.extend([item["text"] for item in current_row])

            current_row = [line]
            current_row_y = line["y"]

    if current_row:
        current_row.sort(key=lambda k: k["x"])
        sorted_lines.extend([item["text"] for item in current_row])

    return "\n".join(sorted_lines)

def get_clean_text_from_xml(xml_bytes: bytes) -> str:
    """
    Парсит XML и извлекает очищенный, структурированный текст.
    """
    try:
        root = ET.fromstring(xml_bytes)
        lines_data = extract_raw_lines_from_xml(root, MIN_CONFIDENCE)
        
        if lines_data:
            return sort_and_group_lines(lines_data)

        fallback_text = [
            elem.text.strip() for elem in root.iter() 
            if elem.text and elem.text.strip()
        ]
        return "\n".join(fallback_text)
        
    except ET.ParseError:
        return "Error: Could not parse XML content."

def process_single_xml(xml_path: Path):
    """
    Обрабатывает отдельный XML-файл и сохраняет результат в файл с суффиксом -clean.
    """
    clean_filename = xml_path.name.replace("-xml", "-clean")
    clean_path = xml_path.parent / clean_filename

    try:
        with open(xml_path, "rb") as f:
            xml_data = f.read()

        clean_text = get_clean_text_from_xml(xml_data)

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"Processed: {clean_filename}")

    except Exception as e:
        print(f"Error processing {xml_path.name}: {e}")

def main():
    """
    Основной цикл обработки всех найденных XML-файлов в директории.
    """
    base_path = Path.cwd() / SOURCE_DIR
    if not base_path.exists():
        print(f"Error: Directory '{SOURCE_DIR}' not found.")
        return

    xml_files = list(base_path.glob("*-xml"))
    if not xml_files:
        print(f"No *-xml files found in {SOURCE_DIR}")
        return

    print(f"Processing {len(xml_files)} XML files...")
    for xml_path in xml_files:
        process_single_xml(xml_path)

if __name__ == "__main__":
    main()
