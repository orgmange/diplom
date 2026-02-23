import pytest
import xml.etree.ElementTree as ET
from app.services.cleaner_service import CleanerService

@pytest.fixture
def cleaner():
    return CleanerService(min_confidence=50)

def test_extract_raw_lines(cleaner):
    xml_content = """
    <root>
        <RIL_TEXTLINE type="RIL_TEXTLINE" Y="100" X="50" cnf="80">
            <RIL_WORD type="RIL_WORD">Hello</RIL_WORD>
            <RIL_WORD type="RIL_WORD">World</RIL_WORD>
        </RIL_TEXTLINE>
        <RIL_TEXTLINE type="RIL_TEXTLINE" Y="100" X="200" cnf="40">
            <RIL_WORD type="RIL_WORD">Ignored</RIL_WORD>
        </RIL_TEXTLINE>
    </root>
    """
    root = ET.fromstring(xml_content)
    lines = cleaner._extract_raw_lines(root)
    
    assert len(lines) == 1
    assert lines[0]["text"] == "Hello World"
    assert lines[0]["y"] == 100

def test_sort_and_group_lines(cleaner):
    lines_data = [
        {"y": 105, "x": 200, "text": "Right"},
        {"y": 100, "x": 50, "text": "Left"},
        {"y": 200, "x": 50, "text": "Second Row"}
    ]
    
    result = cleaner._sort_and_group_lines(lines_data, threshold=25)
    
    # Current implementation joins sorted items with \n
    assert "Left" in result
    assert "Right" in result
    assert "Second Row" in result
    # It should have 3 lines because they are added individually in the loop
    assert len(result.split("\n")) == 3

def test_parse_xml_bytes_empty(cleaner):
    assert cleaner.parse_xml_bytes(b"invalid xml") == ""
