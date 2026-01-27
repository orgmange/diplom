import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = os.path.join("data", "ocr")
MIN_CONFIDENCE = 0


def get_clean_text_from_xml(xml_bytes: bytes) -> str:
    """Parses XML, sorts lines spatially, filters by confidence, and extracts text."""
    try:
        root = ET.fromstring(xml_bytes)

        lines_data = []

        # Search for any node with type="RIL_TEXTLINE"
        text_lines = [
            elem for elem in root.iter() if elem.get("type") == "RIL_TEXTLINE"
        ]

        if text_lines:
            for line_node in text_lines:
                # Check confidence
                try:
                    cnf = int(line_node.get("cnf", 100))  # Default to 100 if missing
                except ValueError:
                    cnf = 100

                if cnf < MIN_CONFIDENCE:
                    continue

                # Get coordinates
                try:
                    y = int(line_node.get("Y", 0))
                    x = int(line_node.get("X", 0))
                    h = int(line_node.get("H", 10))
                except ValueError:
                    y, x, h = 0, 0, 10

                # Find all RIL_WORD children
                words = []
                for word_node in line_node.iter():
                    if (
                        word_node.get("type") == "RIL_WORD"
                        and word_node.text
                        and word_node.text.strip()
                    ):
                        words.append(word_node.text.strip())

                if words:
                    text = " ".join(words)
                    lines_data.append({"y": y, "x": x, "h": h, "text": text})

            # Sort logic
            lines_data.sort(key=lambda k: k["y"])

            sorted_lines = []
            current_row = []
            current_row_y = -1000

            for line in lines_data:
                # Threshold for visual row
                threshold = 25

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

        # Fallback if no lines found (and no confidence check possible easily)
        text_content = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_content.append(elem.text.strip())
        return "\n".join(text_content)
    except ET.ParseError:
        return "Error: Could not parse XML content."


def main():
    base_path = Path.cwd() / SOURCE_DIR
    if not base_path.exists():
        print(f"Error: Directory '{SOURCE_DIR}' not found.")
        return

    # Find all *-xml files
    xml_files = list(base_path.glob("*-xml"))

    if not xml_files:
        print(f"No *-xml files found in {SOURCE_DIR}")
        return

    print(
        f"Found {len(xml_files)} XML files. Processing with MIN_CONFIDENCE={MIN_CONFIDENCE}..."
    )

    for xml_path in xml_files:
        # Determine output filename: remove '-xml' and add '-clean'
        # e.g. "image.jpg-xml" -> "image.jpg-clean"
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

    print("\nDone.")


if __name__ == "__main__":
    main()
