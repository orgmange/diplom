import json
import os
import re
import string
from pathlib import Path
from datetime import datetime

# Настройки
BASE_DIR = Path(__file__).parent.parent
REFS_DIR = BASE_DIR / "data/references"
OUTPUT_DIR = BASE_DIR / "tests/model_outputs"
REPORTS_DIR = BASE_DIR / "reports"

def normalize_string(s):
    """Приводит строку к нижнему регистру, удаляет пробелы и знаки препинания."""
    if s is None:
        return ""
    s = str(s).lower()
    # Удаляем знаки препинания
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    # Удаляем все пробельные символы
    s = re.sub(r"\s+", "", s)
    return s

def flatten_dict(d, parent_key='', sep='_'):
    """Преобразует вложенный словарь в плоский."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_metrics(reference, candidate):
    """Считает метрики и генерирует детальное сравнение полей."""
    ref_flat = flatten_dict(reference)
    cand_flat = flatten_dict(candidate)
    
    all_keys = set(ref_flat.keys()) | set(cand_flat.keys())
    tp = 0 
    fp = 0 
    fn = 0 
    
    comparison = []

    for key in sorted(all_keys):
        ref_raw = ref_flat.get(key)
        cand_raw = cand_flat.get(key)
        
        ref_val = normalize_string(ref_raw)
        cand_val = normalize_string(cand_raw)
        
        status = "unknown"
        match_type = "neutral" # for coloring: match, mismatch, extra, missing

        if ref_val and cand_val:
            if ref_val == cand_val:
                tp += 1
                status = "Match"
                match_type = "match"
            else:
                fp += 1
                fn += 1
                status = "Mismatch"
                match_type = "mismatch"
        elif ref_val and not cand_val:
            fn += 1
            status = "Missing in Candidate"
            match_type = "missing"
        elif not ref_val and cand_val:
            fp += 1
            status = "Extra in Candidate"
            match_type = "extra"
        else:
             # Both are empty/falsy -> This is a correct match for "empty"
             tp += 1
             status = "Match (Empty)"
             match_type = "match"

        comparison.append({
            "key": key,
            "ref": str(ref_raw) if ref_raw is not None else "—",
            "cand": str(cand_raw) if cand_raw is not None else "—",
            "status": status,
            "type": match_type
        })

    total_keys = len(all_keys)
    accuracy = tp / total_keys if total_keys > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "comparison": comparison
    }

def generate_html_report(all_results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    model_stats = {}
    doc_names = sorted(list(all_results.keys()))
    
    # Collect data for charts and summary
    flat_results = []

    for doc, res_list in all_results.items():
        for res in res_list:
            m_name = res['model']
            if m_name not in model_stats:
                model_stats[m_name] = {"accs": [], "times": []}
            model_stats[m_name]["accs"].append(res['metrics']['accuracy'])
            model_stats[m_name]["times"].append(res.get('elapsed', 0))
            
            flat_results.append({
                "model": m_name,
                "document": doc,
                "accuracy": res['metrics']['accuracy'],
                "precision": res['metrics']['precision'],
                "recall": res['metrics']['recall'],
                "f1": res['metrics']['f1'],
                "time": res.get('elapsed', 0)
            })

    # Prepare Chart Data
    labels = doc_names
    datasets_acc = []
    avg_times = []
    model_labels = []
    
    colors = [
        'rgba(255, 99, 132, 0.7)', 'rgba(54, 162, 235, 0.7)', 
        'rgba(255, 206, 86, 0.7)', 'rgba(75, 192, 192, 0.7)', 
        'rgba(153, 102, 255, 0.7)', 'rgba(255, 159, 64, 0.7)',
        'rgba(199, 199, 199, 0.7)'
    ]
    
    i = 0
    for m_name, stats in model_stats.items():
        color = colors[i % len(colors)]
        datasets_acc.append({
            "label": m_name,
            "data": stats["accs"],
            "backgroundColor": color
        })
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        avg_times.append(avg_time)
        model_labels.append(m_name)
        i += 1

    html = f"""
    <html>
    <head>
        <title>OCR Model Benchmark Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f4f4f9; color: #333; }}
            h1, h2, h3 {{ color: #444; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .chart-container {{ position: relative; height: 350px; width: 100%; margin-bottom: 30px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #ddd; vertical-align: top; }}
            th {{ background-color: #f8f9fa; font-weight: 600; }}
            tr:last-child td {{ border-bottom: none; }}
            .metric {{ font-weight: bold; color: #2a52be; }}
            
            /* Comparison Table Styles */
            .match {{ background-color: #d4edda; color: #155724; }}
            .mismatch {{ background-color: #f8d7da; color: #721c24; }}
            .missing {{ background-color: #fff3cd; color: #856404; }}
            .extra {{ background-color: #f8d7da; color: #721c24; }}
            .neutral {{ color: #666; }}
            
            details summary {{ cursor: pointer; color: #007bff; font-weight: 500; outline: none; }}
            details summary:hover {{ text-decoration: underline; }}
            .comp-table {{ width: 100%; font-size: 0.9em; margin-top: 10px; border: 1px solid #eee; }}
            .comp-table th {{ background: #eee; font-size: 0.85em; }}
            .comp-table td {{ padding: 6px; border: 1px solid #eee; word-break: break-all; }}
            
            .summary-section {{ margin-top: 50px; padding-top: 20px; border-top: 2px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Benchmark Report - {timestamp}</h1>

            <div class="chart-container">
                <canvas id="accChart"></canvas>
            </div>
            
            <div class="chart-container">
                <canvas id="timeChart"></canvas>
            </div>

            <hr>
    """
    
    # Per Document Section
    for doc_name in doc_names:
        results = all_results[doc_name]
        html += f"<h2>Document: {doc_name}</h2>"
        html += """
        <table>
            <tr>
                <th style="width: 15%">Model</th>
                <th style="width: 8%">Acc</th>
                <th style="width: 8%">Prec</th>
                <th style="width: 8%">Rec</th>
                <th style="width: 8%">F1</th>
                <th style="width: 8%">Time (s)</th>
                <th>Field Analysis</th>
            </tr>
        """
        for res in results:
            m = res['metrics']
            
            comp_rows = ""
            for row in m['comparison']:
                comp_rows += f"""
                <tr class="{row['type']}">
                    <td>{row['key']}</td>
                    <td>{row['ref']}</td>
                    <td>{row['cand']}</td>
                    <td>{row['status']}</td>
                </tr>
                """

            html += f"""
            <tr>
                <td><strong>{res['model']}</strong></td>
                <td class="metric">{m['accuracy']}</td>
                <td>{m['precision']}</td>
                <td>{m['recall']}</td>
                <td>{m['f1']}</td>
                <td>{res.get('elapsed', 0)}</td>
                <td>
                    <details>
                        <summary>Show Comparison</summary>
                        <table class="comp-table">
                            <thead>
                                <tr>
                                    <th style="width: 25%">Key</th>
                                    <th style="width: 30%">Reference</th>
                                    <th style="width: 30%">Candidate</th>
                                    <th style="width: 15%">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {comp_rows}
                            </tbody>
                        </table>
                    </details>
                </td>
            </tr>
            """
        html += "</table>"

    # Global Summary Section
    html += """
        <div class="summary-section">
            <h2>Global Summary (Average Across All Documents)</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Avg Accuracy</th>
                    <th>Avg Precision</th>
                    <th>Avg Recall</th>
                    <th>Avg F1-Score</th>
                    <th>Avg Time (s)</th>
                </tr>
    """
    
    # Aggregate results by model
    model_aggregates = {}
    for res in flat_results:
        m_name = res['model']
        if m_name not in model_aggregates:
            model_aggregates[m_name] = {
                "count": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "time": 0.0
            }
        
        agg = model_aggregates[m_name]
        agg["count"] += 1
        agg["accuracy"] += res['accuracy']
        agg["precision"] += res['precision']
        agg["recall"] += res['recall']
        agg["f1"] += res['f1']
        agg["time"] += res['time']

    # Calculate averages
    summary_list = []
    for m_name, agg in model_aggregates.items():
        count = agg["count"]
        if count > 0:
            summary_list.append({
                "model": m_name,
                "avg_accuracy": round(agg["accuracy"] / count, 2),
                "avg_precision": round(agg["precision"] / count, 2),
                "avg_recall": round(agg["recall"] / count, 2),
                "avg_f1": round(agg["f1"] / count, 2),
                "avg_time": round(agg["time"] / count, 2)
            })

    # Sort by Avg Accuracy DESC
    sorted_summary = sorted(summary_list, key=lambda x: x['avg_accuracy'], reverse=True)
    
    for idx, row in enumerate(sorted_summary, 1):
        html += f"""
        <tr>
            <td>{idx}</td>
            <td>{row['model']}</td>
            <td class="metric">{row['avg_accuracy']}</td>
            <td>{row['avg_precision']}</td>
            <td>{row['avg_recall']}</td>
            <td>{row['avg_f1']}</td>
            <td>{row['avg_time']}</td>
        </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    html += f"""
        </div>
        <script>
            const ctxAcc = document.getElementById('accChart').getContext('2d');
            new Chart(ctxAcc, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: {json.dumps(datasets_acc)}
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{ display: true, text: 'Accuracy by Model per Document' }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true, max: 1.0 }}
                    }}
                }}
            }});

            const ctxTime = document.getElementById('timeChart').getContext('2d');
            new Chart(ctxTime, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(model_labels)},
                    datasets: [{{
                        label: 'Average Execution Time (s)',
                        data: {json.dumps(avg_times)},
                        backgroundColor: 'rgba(75, 192, 192, 0.7)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{ display: true, text: 'Average Execution Time per Model (Seconds)' }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    report_path = REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path

def main():
    if not OUTPUT_DIR.exists():
        print("Error: Output directory missing. Run inference script first.")
        return

    # Load all references first
    references = {}
    if not REFS_DIR.exists():
         print(f"Error: References directory {REFS_DIR} not found.")
         return

    for ref_file in REFS_DIR.glob("*-reference.json"):
        doc_name = ref_file.name.replace("-reference.json", "")
        with open(ref_file, "r", encoding="utf-8") as f:
            references[doc_name] = json.load(f)

    all_results = {}
    
    # Load results from the latest aggregated file
    result_files = sorted(OUTPUT_DIR.glob("inference_results_*.json"), key=os.path.getmtime, reverse=True)
    if not result_files:
        print("Error: No inference_results_*.json found. Run inference script first.")
        return
    
    latest_results_file = result_files[0]
    print(f"Loading results from: {latest_results_file.name}")
    
    with open(latest_results_file, "r", encoding="utf-8") as f:
        try:
            results_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON results file.")
            return

    for data in results_data:
        doc_name = data.get("doc_name")
        if not doc_name or doc_name not in references:
             # Fallback or skip if doc_name is missing/unknown (shouldn't happen with new script)
             continue
             
        model_name = data.get("model", "unknown")
        parsed_json = data.get("parsed_json", {})
        
        metrics_data = calculate_metrics(references[doc_name], parsed_json)
        
        res_entry = {
            "model": model_name,
            "metrics": metrics_data,
            "elapsed": data.get("elapsed", 0)
        }

        if doc_name not in all_results:
            all_results[doc_name] = []
        all_results[doc_name].append(res_entry)

    if not all_results:
        print("No results matched with references.")
        return

    report_path = generate_html_report(all_results)
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()
