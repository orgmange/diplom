import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.endpoints import ocr_service, cleaner_service, vector_service

client = TestClient(app)

@pytest.fixture
def mock_services(mocker):
    # Mocking OCRService
    mocker.patch.object(ocr_service, 'process_directory', return_value=["test.jpg"])
    # Mocking CleanerService
    mocker.patch.object(cleaner_service, 'process_directory', return_value=["test.jpg-clean"])
    # Mocking VectorService
    mocker.patch.object(vector_service, 'index_directory', return_value=1)
    mocker.patch.object(vector_service, 'search', return_value=[{"filename": "test.jpg-clean", "text": "test document", "score": 0.95}])
    # Mock status check for status endpoint
    mocker.patch('os.path.exists', return_value=True)

def test_scan_ocr(mock_services):
    response = client.post("/api/v1/ocr/scan")
    assert response.status_code == 200
    assert response.json() == {"processed_files": ["test.jpg"]}

def test_clean_ocr(mock_services):
    response = client.post("/api/v1/ocr/clean")
    assert response.status_code == 200
    assert response.json() == {"cleaned_files": ["test.jpg-clean"]}

def test_index_rag(mock_services):
    response = client.post("/api/v1/rag/index")
    assert response.status_code == 200
    assert response.json() == {"indexed_count": 1}

def test_search_rag(mock_services):
    response = client.post("/api/v1/rag/search", json={"query": "test query", "limit": 5})
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["filename"] == "test.jpg-clean"

def test_status(mock_services):
    # Directly mock the return of the function since it's hard to mock all glob calls
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert "ocr_files" in response.json()
    assert "vectorized_count" in response.json()
