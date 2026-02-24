from pathlib import Path

from app.services.benchmark_service import BenchmarkService


def test_run_benchmark_pipeline(tmp_path, mocker):
    template_dir = tmp_path / "ocr"
    docs_dir = tmp_path / "docs"
    passport_dir = docs_dir / "passport"
    xml_dir = passport_dir / "xml"
    clean_dir = passport_dir / "clean"
    template_dir.mkdir()
    xml_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    (template_dir / "passport2.jpg-xml").write_text("raw passport", encoding="utf-8")
    (template_dir / "passport2.jpg-clean").write_text("clean passport", encoding="utf-8")
    (xml_dir / "паспорт мой.png-xml").write_text("raw test", encoding="utf-8")
    (clean_dir / "паспорт мой.png-clean").write_text("clean test", encoding="utf-8")

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
    mocker.patch("app.services.benchmark_service.settings.DOCS_DIR", docs_dir)
    report = benchmark.run("nomic-embed-text:latest")

    assert report["embedding_model"] == "nomic-embed-text:latest"
    assert report["prepared"]["prepared_xml"] == 0
    assert report["prepared"]["prepared_clean"] == 0
    assert report["indexed"]["total_count"] == 2
    assert report["raw_tests"]["correct"] == 1
    assert report["clean_tests"]["correct"] == 1
    assert report["overall"]["total"] == 2
    assert report["overall"]["correct"] == 2
    assert report["overall"]["accuracy"] == 1.0
    service.reset_collection.assert_called_once_with(vector_size=1024)


def test_prepare_test_corpus_from_doc_folders(tmp_path, mocker):
    docs_dir = tmp_path / "docs"
    image_dir = docs_dir / "passport" / "image"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.jpg"
    image_path.write_bytes(b"image")

    vector_service = mocker.Mock()
    ocr_service = mocker.Mock()
    cleaner_service = mocker.Mock()
    ocr_service.create_task.return_value = "task-id"
    ocr_service.wait_for_task.return_value = "success"
    ocr_service.fetch_result.return_value = b"<xml></xml>"
    cleaner_service.parse_xml_bytes.return_value = "clean text"

    benchmark = BenchmarkService(
        vector_service=vector_service,
        ocr_service=ocr_service,
        cleaner_service=cleaner_service,
    )
    mocker.patch("app.services.benchmark_service.settings.DOCS_DIR", docs_dir)

    result = benchmark._prepare_test_corpus(docs_dir)

    assert result == {"prepared_xml": 1, "prepared_clean": 1}
    assert (docs_dir / "passport" / "xml" / "sample.jpg-xml").exists()
    assert (docs_dir / "passport" / "clean" / "sample.jpg-clean").exists()
