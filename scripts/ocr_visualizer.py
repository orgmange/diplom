import os
import html
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_DIR = "документы для тестирвания моделей"
# Supported image extensions to look for pairing
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def create_html_view(xml_path: Path, image_path: Path):
    try:
        with open(xml_path, "rb") as f:
            xml_data = f.read()
        root = ET.fromstring(xml_data)
    except Exception as e:
        print(f"Error reading {xml_path.name}: {e}")
        return

    # Extract Page dimensions
    # Usually in the first node or the root if it's RIL_PAGE
    # The example XML showed <node type="RIL_PAGE" W="1200" H="1650">
    # We look for the first RIL_PAGE
    page_node = None
    for elem in root.iter():
        if elem.get('type') == 'RIL_PAGE':
            page_node = elem
            break
    
    # Default size if not found
    width = 1000
    height = 1000
    
    if page_node is not None:
        width = int(page_node.get('W', width))
        height = int(page_node.get('H', height))

    boxes_html = ""
    
    # Iterate over text lines
    for elem in root.iter():
        if elem.get('type') == 'RIL_TEXTLINE':
            try:
                x = int(elem.get('X', 0))
                y = int(elem.get('Y', 0))
                w = int(elem.get('W', 0))
                h = int(elem.get('H', 0))
                cnf = int(elem.get('cnf', 0))
                
                # Extract text
                words = []
                for word_node in elem.iter():
                    if word_node.get('type') == 'RIL_WORD' and word_node.text:
                        words.append(word_node.text)
                text = " ".join(words)
                
                # Color coding based on confidence
                if cnf >= 90:
                    border_color = "rgba(0, 255, 0, 0.7)" # Green
                    bg_color = "rgba(0, 255, 0, 0.1)"
                elif cnf >= 60:
                    border_color = "rgba(255, 165, 0, 0.7)" # Orange
                    bg_color = "rgba(255, 165, 0, 0.1)"
                else:
                    border_color = "rgba(255, 0, 0, 0.8)" # Red
                    bg_color = "rgba(255, 0, 0, 0.1)"

                safe_text = html.escape(text)
                
                boxes_html += f'''
                <div class="box" style="left: {x}px; top: {y}px; width: {w}px; height: {h}px; border-color: {border_color}; background-color: {bg_color};" title="Text: {safe_text}&#10;Conf: {cnf}%">
                    <span class="label">{cnf}%</span>
                </div>
                '''
            except ValueError:
                continue

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OCR View: {image_path.name}</title>
    <style>
        body {{ background-color: #333; color: white; font-family: sans-serif; padding: 20px; }}
        .container {{ position: relative; display: inline-block; }}
        .ocr-image {{ display: block; max-width: 100%; border: 1px solid #555; }}
        .box {{
            position: absolute;
            border: 2px solid;
            box-sizing: border-box;
            cursor: help;
        }}
        .box:hover {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            z-index: 10;
        }}
        .label {{
            position: absolute;
            top: -14px;
            right: 0;
            background: rgba(0,0,0,0.7);
            color: #fff;
            font-size: 10px;
            padding: 1px 3px;
            pointer-events: none;
        }}
        .info {{ margin-bottom: 10px; }}
        .legend {{ margin-top: 10px; font-size: 0.9em; }}
        .l-green {{ color: #0f0; }} .l-orange {{ color: orange; }} .l-red {{ color: #f55; }}
    </style>
</head>
<body>
    <div class="info">
        <h1>{image_path.name}</h1>
        <p>Hover over boxes to see detected text.</p>
    </div>
    <div class="container" style="width: {width}px; height: {height}px;">
        <img src="{image_path.name}" class="ocr-image" width="{width}" height="{height}">
        {boxes_html}
    </div>
    <div class="legend">
        Legend: 
        <span class="l-green">Green >= 90%</span> | 
        <span class="l-orange">Orange >= 60%</span> | 
        <span class="l-red">Red < 60%</span>
    </div>
</body>
</html>'''

    output_html_path = image_path.parent / f"{image_path.name}-view.html"
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Created: {output_html_path.name}")

def main():
    base_path = Path.cwd() / SOURCE_DIR
    if not base_path.exists():
        print(f"Error: Directory '{SOURCE_DIR}' not found.")
        return

    xml_files = list(base_path.glob("*-xml"))
    if not xml_files:
        print("No XML files found.")
        return

    print(f"Generating HTML views for {len(xml_files)} files...")

    for xml_path in xml_files:
        # Find corresponding image
        # xml name is "image.ext-xml" -> image name is "image.ext"
        image_name = xml_path.name.replace("-xml", "")
        image_path = xml_path.parent / image_name
        
        if image_path.exists():
            create_html_view(xml_path, image_path)
        else:
            print(f"Warning: Image not found for {xml_path.name}")

    print("\nDone. Open the *-view.html files in your browser.")

if __name__ == "__main__":
    main()
