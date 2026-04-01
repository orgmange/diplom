import pytest
from unittest.mock import MagicMock, patch
from app.api.endpoints import (
    SearchQuery,
    get_status,
    index_documents,
    list_benchmark_models,
    run_benchmark,
    scan_ocr,
    clean_ocr,
    vector_service,
    ocr_service,
    cleaner_service,
    benchmark_service
)
from app.api.external import RecognitionRequest
from app.services.recognition_service import RecognitionService
from app.core.config import settings

# --- API Endpoints Tests ---

def test_index_documents(mocker):
    mocker.patch.object(vector_service, "index_examples", return_value=["doc1.xml"])
    assert index_documents() == {"indexed_files": ["doc1.xml"], "count": 1}

def test_search_documents(mocker):
    mocker.patch.object(
        vector_service,
        "search",
        return_value=[{"filename": "doc1.xml", "score": 0.95}],
    )
    payload = SearchQuery(query="test", limit=1)
    from app.api.endpoints import search_documents
    result = search_documents(payload)
    assert result[0]["filename"] == "doc1.xml"

def test_list_benchmark_models(mocker):
    mocker.patch.object(vector_service, "list_embedding_models", return_value=["model1"])
    assert list_benchmark_models() == {"models": ["model1"]}

def test_run_benchmark(mocker):
    mock_result = {
        "embedding_model": "model1",
        "indexed": {"total_count": 1, "files": ["f1"]},
        "results": {"total": 1, "correct": 1, "accuracy": 1.0}
    }
    mocker.patch.object(benchmark_service, "run", return_value=mock_result)
    from app.api.endpoints import BenchmarkRunRequest
    response = run_benchmark(BenchmarkRunRequest(embedding_model="model1"))
    assert response["embedding_model"] == "model1"

def test_status(mocker):
    mock_count = MagicMock()
    mock_count.count = 5
    mocker.patch.object(vector_service.client, "count", return_value=mock_count)
    mocker.patch.object(vector_service, "_load_state", return_value="model1")
    response = get_status()
    assert response["vectorized_count"] == 5
    assert response["current_model"] == "model1"

def test_scan_ocr(mocker):
    mocker.patch.object(ocr_service, "process_docs_directory", return_value=["p1.jpg"])
    assert scan_ocr() == {"processed_files": ["p1.jpg"]}

def test_clean_ocr(mocker):
    mocker.patch.object(
        cleaner_service,
        "process_docs_directory",
        return_value=[{"filename": "p1-clean", "snippet": "text"}],
    )
    assert clean_ocr() == {"cleaned_files": [{"filename": "p1-clean", "snippet": "text"}]}

# --- Recognition Service Tests ---

@pytest.mark.asyncio
async def test_recognition_flow(mocker):
    # Mocking dependencies
    mock_ocr = mocker.patch("app.services.recognition_service.OCRService")
    mock_cleaner = mocker.patch("app.services.recognition_service.CleanerService")
    # StructuringService.structure is async, use AsyncMock
    mock_structuring = mocker.patch("app.services.recognition_service.StructuringService")
    mock_structuring.return_value.structure = mocker.AsyncMock(return_value={"result": {"structured": "data"}})
    
    service = RecognitionService(
        ocr_service=mock_ocr.return_value,
        cleaner_service=mock_cleaner.return_value,
        structuring_service=mock_structuring.return_value
    )
    
    mock_ocr.return_value.create_task.return_value = "task123"
    mock_ocr.return_value.wait_for_task.return_value = "success"
    mock_ocr.return_value.fetch_result.return_value = b"<xml>raw</xml>"
    mock_cleaner.return_value.parse_xml_bytes.return_value = "cleaned text"
    
    request = RecognitionRequest(image_base64="YmFzZTY0")
    task_id = service.start_recognition_task(request.image_base64)
    
    assert task_id is not None
    
    import asyncio
    await asyncio.sleep(0.1) # Give it some time to run
    
    result = service.get_task_result(task_id)
    assert result is not None
    if result["status"] == "error":
        print(f"Error: {result.get('error')}")
    assert result["status"] == "completed"
    assert result["result"] == {"structured": "data"}
