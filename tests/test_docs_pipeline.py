from app.services.cleaner_service import CleanerService
from app.services.ocr_service import OCRService


def test_ocr_service_process_docs_directory(tmp_path, mocker):
    docs_dir = tmp_path / "docs"
    image_dir = docs_dir / "passport" / "image"
    image_dir.mkdir(parents=True)
    (image_dir / "a.jpg").write_bytes(b"img")

    service = OCRService()
    mocker.patch("app.services.ocr_service.settings.DOCS_DIR", docs_dir)
    mocker.patch.object(service, "create_task", return_value="task-id")
    mocker.patch.object(service, "wait_for_task", return_value="success")
    mocker.patch.object(service, "fetch_result", return_value=b"<xml></xml>")

    processed = service.process_docs_directory()

    assert processed == ["passport/a.jpg"]
    assert (docs_dir / "passport" / "xml" / "a.jpg-xml").exists()


def test_cleaner_service_process_docs_directory(tmp_path, mocker):
    docs_dir = tmp_path / "docs"
    xml_dir = docs_dir / "passport" / "xml"
    xml_dir.mkdir(parents=True)
    (xml_dir / "a.jpg-xml").write_bytes(b"<root><RIL_WORD type='RIL_WORD'>test</RIL_WORD></root>")

    service = CleanerService()
    mocker.patch("app.services.cleaner_service.settings.DOCS_DIR", docs_dir)

    cleaned = service.process_docs_directory()

    assert len(cleaned) == 1
    assert cleaned[0]["filename"] == "passport/a.jpg-clean"
    assert (docs_dir / "passport" / "clean" / "a.jpg-clean").exists()
