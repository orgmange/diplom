import json
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
    # name matches (1), mrz doesn't match perfectly (0) -> 1/2 = 0.5
    assert accuracy == 0.5

def test_calculate_accuracy_normalization():
    service = StructuringBenchmarkService(None)
    reference = {"name": "IVAN"}
    result = {"name": " ivan "}
    
    accuracy = service._calculate_accuracy(result, reference)
    assert accuracy == 1.0

def test_run_benchmark_mocked(tmp_path, mocker):
    docs_dir = tmp_path / "docs"
    ref_dir = tmp_path / "benchmark" / "references"
    
    passport_dir = docs_dir / "passport"
    clean_dir = passport_dir / "clean"
    clean_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    
    (clean_dir / "test1.jpg-clean").write_text("cleaned text", encoding="utf-8")
    (ref_dir / "test1.jpg-reference.json").write_text(json.dumps({"name": "Ivan"}), encoding="utf-8")
    
    struct_service = mocker.Mock()
    struct_service.structure.return_value = {
        "result": {"name": "Ivan"},
        "doc_type": "passport"
    }
    
    mocker.patch("app.services.structuring_benchmark_service.settings.DOCS_DIR", docs_dir)
    mocker.patch("app.services.structuring_benchmark_service.settings.BENCHMARK_REF_DIR", ref_dir)
    
    service = StructuringBenchmarkService(struct_service)
    report = service.run(model_name="test-model")
    
    assert report.total_files == 1
    assert report.files_with_reference == 1
    assert report.avg_accuracy == 1.0
    assert report.items[0].filename == "test1.jpg"
    assert report.items[0].expected_type == "passport"
    assert report.items[0].is_type_correct is True
    assert report.items[0].accuracy == 1.0
