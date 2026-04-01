import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from app.core.config import settings

class CleanerService:
    """
    Сервис для очистки и структурирования текста из XML-результатов OCR.
    """
    def __init__(self, min_confidence: int = 0):
        self.min_confidence = min_confidence

    def _extract_raw_lines(self, root: ET.Element) -> List[Dict]:
        """Извлекает строки с координатами и уверенностью."""
        lines_data = []
        text_lines = [elem for elem in root.iter() if elem.get("type") == "RIL_TEXTLINE"]

        for line_node in text_lines:
            try:
                cnf = int(line_node.get("cnf", 100))
            except ValueError:
                cnf = 100
            
            if cnf < self.min_confidence:
                continue

            try:
                y = int(line_node.get("Y", 0))
                x = int(line_node.get("X", 0))
            except ValueError:
                y, x = 0, 0

            words = [
                w.text.strip() for w in line_node.iter() 
                if w.get("type") == "RIL_WORD" and w.text and w.text.strip()
            ]

            if words:
                lines_data.append({"y": y, "x": x, "text": " ".join(words)})
        
        return lines_data

    def _sort_and_group_lines(self, lines_data: List[Dict], threshold: int = 25) -> str:
        """Сортирует строки по Y, группирует близкие строки, затем сортирует по X."""
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

    def parse_xml_bytes(self, xml_bytes: bytes) -> str:
        """Парсит XML байты и возвращает очищенный текст."""
        try:
            root = ET.fromstring(xml_bytes)
            lines_data = self._extract_raw_lines(root)
            if lines_data:
                return self._sort_and_group_lines(lines_data)
            
            # Fallback
            return "\n".join([
                elem.text.strip() for elem in root.iter() 
                if elem.text and elem.text.strip()
            ])
        except ET.ParseError:
            return ""

    def _iter_doc_dirs(self) -> List[Path]:
        if not settings.DOCS_DIR.exists():
            return []
        return sorted(path for path in settings.DOCS_DIR.iterdir() if path.is_dir())

    def process_docs_directory(self) -> List[Dict[str, str]]:
        """
        Сканирует data/docs/*/xml и создает data/docs/*/clean.
        Возвращает список объектов с информацией об обработанных файлах.
        """
        processed: List[Dict[str, str]] = []
        for doc_dir in self._iter_doc_dirs():
            xml_dir = doc_dir / "xml"
            clean_dir = doc_dir / "clean"
            if not xml_dir.exists():
                continue
            clean_dir.mkdir(parents=True, exist_ok=True)
            xml_files = sorted(path for path in xml_dir.iterdir() if path.is_file())
            for xml_path in xml_files:
                clean_filename = xml_path.name.replace("-xml", "-clean")
                clean_path = clean_dir / clean_filename
                if clean_path.exists():
                    continue
                try:
                    content = self.parse_xml_bytes(xml_path.read_bytes())
                    if content:
                        clean_path.write_text(content, encoding="utf-8")
                        processed.append(
                            {
                                "filename": f"{doc_dir.name}/{clean_filename}",
                                "snippet": content[:100] + "..." if len(content) > 100 else content,
                            }
                        )
                except Exception as e:
                    print(f"Error cleaning {xml_path.name}: {e}")
        return processed


