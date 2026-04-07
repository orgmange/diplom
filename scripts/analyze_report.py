import json
import sys

report_path = "data/benchmark/reports/report_qwen3.5_9b_20260406_144948.json"

with open(report_path, "r", encoding="utf-8") as f:
    d = json.load(f)

print(f"Total files: {d['total_files']}")
print(f"Files with reference: {d['files_with_reference']}")
print("=" * 80)

for item in d['items']:
    fm = item.get('field_metrics', [])
    ref = item.get('reference_json') or {}
    res = item.get('result_json') or {}

    # Count nested keys in reference
    def count_leaf_keys(obj, prefix=""):
        count = 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    count += count_leaf_keys(v, f"{prefix}{k}.")
                else:
                    count += 1
        return count

    ref_leaf_count = count_leaf_keys(ref)
    
    print(f"File: {item['filename']}")
    print(f"  Doc type: {item['expected_type']} -> {item['detected_type']} (correct: {item['is_type_correct']})")
    print(f"  Reference leaf fields: {ref_leaf_count}")
    print(f"  Fields compared (field_metrics): {len(fm)}")
    if fm:
        print(f"  Fields: {[f['field_name'] for f in fm]}")
    if ref_leaf_count != len(fm):
        print(f"  *** MISMATCH: ref has {ref_leaf_count} leaf fields, but {len(fm)} compared! ***")
    print()
