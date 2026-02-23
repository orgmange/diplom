import pytest
import uuid
from unittest.mock import MagicMock
from app.services.vector_service import VectorService
from qdrant_client.http import models

@pytest.fixture
def vector_service(mocker):
    service = VectorService()
    # Mocking Ollama client
    mock_ollama = MagicMock()
    mock_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mocker.patch.object(service, '_ollama_client', mock_ollama)
    
    # Mocking Qdrant client
    mock_qdrant = MagicMock()
    mocker.patch.object(service, '_client', mock_qdrant)
    
    return service, mock_ollama, mock_qdrant

def test_vectorize_text(vector_service):
    service, mock_ollama, _ = vector_service
    embedding = service.vectorize_text("Hello")
    
    assert embedding == [0.1, 0.2, 0.3]
    mock_ollama.embeddings.assert_called_once_with(model="nomic-embed-text:latest", prompt="Hello")

def test_ensure_collection_not_found(vector_service):
    service, _, mock_qdrant = vector_service
    # Simulation of collection not found error
    mock_qdrant.get_collection.side_effect = Exception("Not found")
    
    service.ensure_collection()
    
    mock_qdrant.create_collection.assert_called_once()
    assert mock_qdrant.create_collection.call_args[1]["collection_name"] == "documents"

def test_search(vector_service):
    service, _, mock_qdrant = vector_service
    
    # Мокаем query_points
    mock_result = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {"filename": "doc1", "raw_text": "Raw content", "cleaned_text": "Clean content"}
    mock_point.score = 0.99
    mock_result.points = [mock_point]
    
    mock_qdrant.query_points.return_value = mock_result
    
    results = service.search("query", limit=5)
    
    assert len(results) == 1
    assert results[0]["filename"] == "doc1"
    assert results[0]["raw_text"] == "Raw content"
    assert results[0]["score"] == 0.99
    mock_qdrant.query_points.assert_called_once()
