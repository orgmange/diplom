from pathlib import Path

from app.services.benchmark_service import BenchmarkService


def test_run_benchmark_pipeline(tmp_path, mocker):
    template_dir = tmp_path / "ocr"
    test_dir = tmp_path / "ocr_backup"
    template_dir.mkdir()
    test_dir.mkdir()
    (template_dir / "passport2.jpg-xml").write_text("raw passport", encoding="utf-8")
    (template_dir / "passport2.jpg-clean").write_text("clean passport", encoding="utf-8")
    (test_dir / "паспорт мой.png-xml").write_text("raw test", encoding="utf-8")
    (test_dir / "паспорт мой.png-clean").write_text("clean test", encoding="utf-8")

    service = mocker.Mock()
    service.get_embedding_size.return_value = 1024
    service.reset_collection.return_value = None
    service.index_templates_by_mode.side_effect = [["passport2.jpg-xml"], ["passport2.jpg-clean"]]
    service.search.side_effect = [
        [
            {
                "filename": "passport2.jpg-xml",
                "doc_type": "passport",
                "score": 0.99,
            }
        ],
        [
            {
                "filename": "passport2.jpg-clean",
                "doc_type": "passport",
                "score": 0.98,
            }
        ],
    ]

    benchmark = BenchmarkService(service)
    mocker.patch("app.services.benchmark_service.settings.OCR_DIR", template_dir)
    mocker.patch("app.services.benchmark_service.settings.DATA_DIR", tmp_path)
    report = benchmark.run("nomic-embed-text:latest")

    assert report["embedding_model"] == "nomic-embed-text:latest"
    assert report["indexed"]["total_count"] == 2
    assert report["raw_tests"]["correct"] == 1
    assert report["clean_tests"]["correct"] == 1
    service.reset_collection.assert_called_once_with(vector_size=1024)
