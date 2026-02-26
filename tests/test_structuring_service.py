import json
import pytest
from unittest.mock import MagicMock
from app.services.structuring_service import StructuringService

@pytest.fixture
def mock_vector_service():
    return MagicMock()

@pytest.fixture
def structuring_service(mock_vector_service):
    return StructuringService(mock_vector_service)

def test_build_prompt(structuring_service):
    target_text = "test text"
    examples = [{"cleaned_text": "ex input", "json_output": '{"key": "val"}'}]
    prompt = structuring_service.build_prompt(target_text, examples)
    
    assert "test text" in prompt
    assert "ex input" in prompt
    assert '{"key": "val"}' in prompt
    assert "### TARGET OCR TEXT:" in prompt

def test_get_schema_for_type_passport(structuring_service, tmp_path, mocker):
    # Мокаем путь к шаблонам
    mocker.patch("app.services.structuring_service.settings.BASE_DIR", tmp_path)
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "passport_ru.json").write_text(json.dumps({
        "surname": "str",
        "mrz": {"line1": "str"}
    }), encoding="utf-8")
    
    schema = structuring_service._get_schema_for_type("passport")
    
    assert schema["type"] == "object"
    assert "surname" in schema["properties"]
    assert schema["properties"]["mrz"]["type"] == "object"
    assert "line1" in schema["properties"]["mrz"]["properties"]

def test_structure_success(structuring_service, mock_vector_service, mocker):
    # Мокаем ответ Ollama
    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = {"response": '{"name": "Ivan"}'}
    structuring_service._ollama_client = mock_ollama
    
    mock_vector_service.search.return_value = []
    mock_vector_service._detect_doc_type.return_value = None
    
    result = structuring_service.structure("raw", "clean", "model")
    
    assert result == {"name": "Ivan"}
    mock_ollama.generate.assert_called_once()

def test_structure_empty_response_fallback(structuring_service, mock_vector_service, mocker):
    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = {"response": ""}
    structuring_service._ollama_client = mock_ollama
    
    mock_vector_service.search.return_value = []
    
    result = structuring_service.structure("raw", "clean", "model")
    
    assert "error" in result
    assert "Empty response from model" in result["error"]

def test_structure_json_extract_fallback(structuring_service, mock_vector_service, mocker):
    # Случай, когда модель вернула текст + JSON, а форматтер не сработал
    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = {"response": 'Text before {"name": "Ivan"} text after'}
    structuring_service._ollama_client = mock_ollama
    
    mock_vector_service.search.return_value = []
    
    result = structuring_service.structure("raw", "clean", "model")
    assert result == {"name": "Ivan"}
