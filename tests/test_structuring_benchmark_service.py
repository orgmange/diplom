import json
import asyncio
from pathlib import Path
from app.services.structuring_benchmark_service import StructuringBenchmarkService

def test_calculate_accuracy_simple():
    service = StructuringBenchmarkService(None)
    reference = {"name": "Ivan", "age": 30}
    result = {"name": "Ivan", "age": 30, "extra": "info"}
    
    accuracy = service._calculate_accuracy(result, reference)
    assert accuracy == 1.0

def test_calculate_accuracy_partial():
    service = StructuringBenchmarkService(None)
    reference = {"name": "Ivan", "age": 30}
    result = {"name": "Ivan", "age": 25}
    
    accuracy = service._calculate_accuracy(result, reference)
    assert accuracy == 0.5

def test_calculate_accuracy_nested():
    service = StructuringBenchmarkService(None)
    reference = {"name": "Ivan", "mrz": {"line1": "abc", "line2": "def"}}
    result = {"name": "Ivan", "mrz": {"line1": "abc", "line2": "wrong"}}
    
    accuracy = service._calculate_accuracy(result, reference)
    assert accuracy == 0.5

def test_calculate_accuracy_normalization():
    service = StructuringBenchmarkService(None)
    reference = {"name": "IVAN"}
    result = {"name": " ivan "}
    
    accuracy = service._calculate_accuracy(result, reference)
    assert accuracy == 1.0

def _setup_benchmark_env(tmp_path, mocker, with_reference=True):
    docs_dir = tmp_path / "docs"
    ref_dir = tmp_path / "benchmark" / "references"
    reports_dir = tmp_path / "data" / "benchmark" / "reports"

    passport_dir = docs_dir / "passport"
    clean_dir = passport_dir / "clean"
    clean_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    (clean_dir / "test1.jpg-clean").write_text("cleaned text", encoding="utf-8")
    if with_reference:
        (ref_dir / "test1.jpg-reference.json").write_text(
            json.dumps({"name": "Ivan"}), encoding="utf-8"
        )

    struct_service = mocker.Mock()
    struct_service.structure = mocker.AsyncMock(return_value={
        "result": {"name": "Ivan"},
        "doc_type": "passport"
    })
    struct_service.vector_service = mocker.Mock()
    struct_service.vector_service.client = mocker.Mock()
    count_mock = mocker.Mock()
    count_mock.count = 5
    struct_service.vector_service.client.count.return_value = count_mock

    mocker.patch("app.services.structuring_benchmark_service.settings.DOCS_DIR", docs_dir)
    mocker.patch("app.services.structuring_benchmark_service.settings.BENCHMARK_REF_DIR", ref_dir)
    mocker.patch("app.services.structuring_benchmark_service.settings.BASE_DIR", tmp_path)

    return struct_service

def test_run_benchmark_mocked(tmp_path, mocker):
    struct_service = _setup_benchmark_env(tmp_path, mocker)

    service = StructuringBenchmarkService(struct_service)
    report = asyncio.get_event_loop().run_until_complete(
        service.run(model_name="test-model")
    )

    assert report.total_files == 1
    assert report.files_with_reference == 1
    assert report.avg_accuracy == 1.0
    assert report.items[0].filename == "test1.jpg"
    assert report.items[0].expected_type == "passport"
    assert report.items[0].is_type_correct is True
    assert report.items[0].accuracy == 1.0

def test_run_multi_benchmark(tmp_path, mocker):
    struct_service = _setup_benchmark_env(tmp_path, mocker)

    service = StructuringBenchmarkService(struct_service)
    reports = asyncio.get_event_loop().run_until_complete(
        service.run_multi(model_names=["model-a", "model-b"], embedding_model="nomic")
    )

    assert len(reports) == 2
    assert reports[0].model_name == "model-a"
    assert reports[1].model_name == "model-b"
    assert reports[0].avg_accuracy == 1.0
    assert reports[1].avg_accuracy == 1.0

def test_run_multi_stop_between_models(tmp_path, mocker):
    struct_service = _setup_benchmark_env(tmp_path, mocker)

    service = StructuringBenchmarkService(struct_service)

    original_run = service.run
    async def run_and_stop(*args, **kwargs):
        result = await original_run(*args, **kwargs)
        service._stop_requested = True
        return result
    mocker.patch.object(service, "run", side_effect=run_and_stop)

    reports = asyncio.get_event_loop().run_until_complete(
        service.run_multi(model_names=["model-a", "model-b", "model-c"])
    )

    assert len(reports) == 1
    assert reports[0].model_name == "model-a"
