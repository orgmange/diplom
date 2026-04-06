import pytest
import uuid
from unittest.mock import MagicMock
from app.services.vector_service import VectorService
from qdrant_client.http import models

@pytest.fixture
def vector_service(mocker):
    service = VectorService()
    mock_ollama = MagicMock()
    mock_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
    # Mocking both if needed, but embed is the primary one now
    mocker.patch.object(service, '_ollama_embed_client', mock_ollama)
    mocker.patch.object(service, '_ollama_client', mock_ollama)

    mock_qdrant = MagicMock()
    # Mock get_collection to return a mock with the "right" size (3) to avoid infinite loop in ensure_collection
    mock_info = MagicMock()
    mock_info.config.params.vectors.size = 3
    mock_qdrant.get_collection.return_value = mock_info
    
    mocker.patch.object(service, '_client', mock_qdrant)
    mocker.patch.object(service, '_load_state', return_value="nomic-embed-text:latest")

    return service, mock_ollama, mock_qdrant

def test_vectorize_text(vector_service):
    service, mock_ollama, _ = vector_service
    embedding = service.vectorize_text("Hello")
    
    assert embedding == [0.1, 0.2, 0.3]
    # Check if embeddings was called on ollama_embed_client
    mock_ollama.embeddings.assert_called_once_with(model="nomic-embed-text:latest", prompt="Hello")

def test_ensure_collection_not_found(vector_service):
    service, _, mock_qdrant = vector_service
    mock_qdrant.get_collection.side_effect = Exception("Not found")
    
    service.ensure_collection()
    
    mock_qdrant.create_collection.assert_called_once()
    assert mock_qdrant.create_collection.call_args[1]["collection_name"] == "documents"

def test_search(vector_service):
    service, _, mock_qdrant = vector_service

    mock_result = MagicMock()
    mock_point = MagicMock()
    mock_point.payload = {
        "text": "Raw content", 
        "json_output": "{}", 
        "doc_type": "passport"
    }
    mock_point.score = 0.99
    mock_result.points = [mock_point]
    
    mock_qdrant.query_points.return_value = mock_result
    
    results = service.search("query", limit=5)
    
    assert len(results) == 1
    assert results[0].get("filename") is None
    assert results[0]["text"] == "Raw content"
    assert results[0]["score"] == 0.99
    mock_qdrant.query_points.assert_called_once()


def test_list_embedding_models(vector_service):
    service, mock_ollama, _ = vector_service
    mock_ollama.list.return_value = {
        "models": [
            {"name": "nomic-embed-text:latest"},
            {"name": "qwen3-embedding:0.6b"},
            {"name": "gemma3:1b"},
        ]
    }
    models = service.list_embedding_models()
    # "nomic-embed-text:latest" contains "embed", "qwen3-embedding:0.6b" contains "embedding"
    assert "nomic-embed-text:latest" in models
    assert "qwen3-embedding:0.6b" in models


def test_reset_collection(vector_service):
    service, _, mock_qdrant = vector_service
    service.reset_collection()
    mock_qdrant.delete_collection.assert_called_once()


def test_get_embedding_size(vector_service):
    service, _, _ = vector_service
    assert service.get_embedding_size() == 3
