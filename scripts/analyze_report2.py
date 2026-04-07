import json, sys

d = json.load(open("data/benchmark/reports/report_qwen3.5_9b_20260406_144948.json"))

# Summary: for each doc type, show count of field_metrics
type_fields = {}
issues = []

for item in d['items']:
    fm = item.get('field_metrics', [])
    ref = item.get('reference_json') or {}
    doc_type = item['expected_type']
    
    # Flatten ref keys
    def flatten(d, prefix=""):
        items = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(flatten(v, key))
            else:
                items.append(key)
        return items
    
    ref_keys = flatten(ref) if ref else []
    fm_names = [f['field_name'] for f in fm]
    
    if doc_type not in type_fields:
        type_fields[doc_type] = []
    
    type_fields[doc_type].append({
        'file': item['filename'],
        'ref_count': len(ref_keys),
        'fm_count': len(fm),
        'ref_keys': sorted(ref_keys),
        'fm_keys': sorted(fm_names),
        'accuracy': item['accuracy']
    })
    
    if len(ref_keys) != len(fm):
        issues.append(f"MISMATCH: {item['filename']} ({doc_type}): ref={len(ref_keys)} vs fm={len(fm)}")
    
    # Check if field names match
    missing_in_fm = set(ref_keys) - set(fm_names)
    extra_in_fm = set(fm_names) - set(ref_keys)
    if missing_in_fm:
        issues.append(f"  Missing from metrics: {missing_in_fm}")
    if extra_in_fm:
        issues.append(f"  Extra in metrics: {extra_in_fm}")

print("=== PER DOC TYPE SUMMARY ===")
for dtype, files in sorted(type_fields.items()):
    print(f"\n{dtype}: {len(files)} files")
    for f in files:
        flag = " *** ISSUE ***" if f['ref_count'] != f['fm_count'] else ""
        print(f"  {f['file']}: ref_fields={f['ref_count']}, compared={f['fm_count']}, accuracy={f['accuracy']:.4f}{flag}")
    # Show reference keys for first file
    print(f"  Example ref keys: {files[0]['ref_keys']}")
    print(f"  Example fm keys:  {files[0]['fm_keys']}")

print("\n=== ISSUES ===")
if issues:
    for issue in issues:
        print(issue)
else:
    print("No issues found!")
