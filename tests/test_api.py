from app.api.endpoints import (
    BenchmarkRunRequest,
    SearchQuery,
    get_status,
    index_documents,
    list_benchmark_models,
    run_benchmark,
    search_documents,
    benchmark_service,
    vector_service,
)


def test_index_documents(mocker):
    mocker.patch.object(vector_service, "index_directory", return_value=["doc-xml"])
    assert index_documents() == {"indexed_files": ["doc-xml"]}


def test_search_documents(mocker):
    mocker.patch.object(
        vector_service,
        "search",
        return_value=[{"filename": "doc-xml", "raw_text": "test", "score": 0.95}],
    )
    payload = SearchQuery(query="test", limit=1)
    result = search_documents(payload)
    assert result[0]["filename"] == "doc-xml"


def test_list_benchmark_models(mocker):
    mocker.patch.object(vector_service, "list_embedding_models", return_value=["nomic-embed-text:latest"])
    assert list_benchmark_models() == {"models": ["nomic-embed-text:latest"]}


def test_run_benchmark(mocker):
    mocker.patch.object(
        benchmark_service,
        "run",
        return_value={
            "embedding_model": "nomic-embed-text:latest",
            "indexed": {
                "raw_count": 1,
                "clean_count": 1,
                "total_count": 2,
                "raw_files": ["raw-xml"],
                "clean_files": ["clean-txt"],
            },
            "raw_tests": {
                "total": 1,
                "correct": 1,
                "accuracy": 1.0,
                "items": [
                    {
                        "filename": "raw-test",
                        "expected_type": "passport",
                        "predicted_type": "passport",
                        "predicted_filename": "raw-xml",
                        "score": 0.99,
                        "is_correct": True,
                    }
                ],
            },
            "clean_tests": {
                "total": 1,
                "correct": 0,
                "accuracy": 0.0,
                "items": [
                    {
                        "filename": "clean-test",
                        "expected_type": "snils",
                        "predicted_type": "passport",
                        "predicted_filename": "clean-txt",
                        "score": 0.55,
                        "is_correct": False,
                    }
                ],
            },
        },
    )
    response = run_benchmark(BenchmarkRunRequest(embedding_model="nomic-embed-text:latest"))
    assert response["embedding_model"] == "nomic-embed-text:latest"


def test_status(mocker):
    mock_count = mocker.Mock()
    mock_count.count = 8
    mocker.patch.object(vector_service.client, "count", return_value=mock_count)
    response = get_status()
    assert "ocr_files" in response
    assert response["vectorized_count"] == 8
