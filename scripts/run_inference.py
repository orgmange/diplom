import os
import json
import requests
import time
from pathlib import Path
from datetime import datetime

# Настройки
OLLAMA_API_URL = "http://192.168.0.20:11434/api/generate"
BASE_DIR = Path(__file__).parent.parent
OCR_DIR = BASE_DIR / "data/ocr"
REFS_DIR = BASE_DIR / "data/references"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "tests/model_outputs"

ITERATIONS = 3
MODELS = [
    "gemma3:1b",
    "gemma3:4b",
    "qwen3:8b",
    "qwen3:4b",
    "qwen3:1.7b",
    "ministral-3:8b",
    "ministral-3:3b",
]

def preload_model(model_name):
    print(f"  Loading model {model_name} into memory...")
    try:
        # Send a dummy request to trigger model loading
        # Using a long timeout here specifically for loading
        requests.post(OLLAMA_API_URL, json={"model": model_name, "prompt": ""}, timeout=600)
    except Exception as e:
        print(f"  Warning: Failed to preload model {model_name}: {e}")

def get_model_response(model_name, prompt, retries=3, delay=5):
    payload = {
        "model": model_name, 
        "prompt": prompt, 
        "stream": False, 
        "format": "json"
    }
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response")
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            return f"Error: {str(e)}"

def cleanup_old_outputs():
    """Deletes old JSON result files from the output directory."""
    print("Cleaning up old output files...")
    for f in OUTPUT_DIR.glob("*.json"):
        try:
            f.unlink()
        except Exception as e:
            print(f"  Failed to delete {f.name}: {e}")

def main():
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0. Cleanup
    cleanup_old_outputs()

    if not REFS_DIR.exists() or not OCR_DIR.exists():
        print("Error: Directories missing.")
        return

    # 1. Сбор задач
    tasks = []
    for ref_file in REFS_DIR.glob("*-reference.json"):
        original_name = ref_file.name.replace("-reference.json", "")
        
        with open(ref_file, "r", encoding="utf-8") as f:
            reference_json = json.load(f)

        # Select template based on filename to avoid leaking ground truth
        lower_name = original_name.lower()
        if "паспорт" in lower_name:
            template_path = TEMPLATES_DIR / "passport_ru.json"
        elif "права" in lower_name:
            template_path = TEMPLATES_DIR / "driver_license_ru.json"
        elif "свидетельство" in lower_name:
            template_path = TEMPLATES_DIR / "birth_certificate_ru.json"
        elif "снилс" in lower_name:
            template_path = TEMPLATES_DIR / "snils.json"
        else:
            template_path = None

        if template_path and template_path.exists():
            with open(template_path, "r", encoding="utf-8") as tf:
                schema_hint = json.dumps(json.load(tf), indent=2, ensure_ascii=False)
        else:
            print(f"Warning: No template found for {original_name}, using reference as schema hint.")
            schema_hint = json.dumps(reference_json, indent=2, ensure_ascii=False)
            
        # Task for Clean Text
        clean_file = OCR_DIR / f"{original_name}-clean"
        if clean_file.exists():
            with open(clean_file, "r", encoding="utf-8") as f:
                tasks.append({
                    "doc_name": original_name,
                    "input_text": f.read(),
                    "input_type": "clean",
                    "schema_hint": schema_hint
                })
        else:
            print(f"Warning: Clean file not found for {ref_file.name}")

        # Task for Raw XML
        xml_file = OCR_DIR / f"{original_name}-xml"
        if xml_file.exists():
            with open(xml_file, "r", encoding="utf-8") as f:
                tasks.append({
                    "doc_name": original_name,
                    "input_text": f.read(),
                    "input_type": "raw",
                    "schema_hint": schema_hint
                })
        else:
            print(f"Warning: XML file not found for {ref_file.name}")

    print(f"Found {len(tasks)} tasks (docs * types).")
    
    all_results = []
    
    # 2. Прогон по моделям
    for model in MODELS:
        print(f"\n--- Testing Model: {model} ---")
        preload_model(model)
        
        for task in tasks:
            doc_name = task['doc_name']
            itype = task['input_type']
            
            for i in range(1, ITERATIONS + 1):
                print(f"  Processing {doc_name} [{itype}] (run {i}/{ITERATIONS})...")
                
                context_desc = "OCR text" if itype == "clean" else "raw OCR XML output"
                prompt = f"""
You are a helpful assistant that parses OCR data into JSON.
Here is the {context_desc} from a document:
---
{task['input_text']}
---

Please extract data and return ONLY a valid JSON object matching this structure:
{task['schema_hint']}

Do not include markdown formatting (```json ... ```). Just the raw JSON.
"""
                
                start_time = time.time()
                response_str = get_model_response(model, prompt)
                elapsed = time.time() - start_time
                
                candidate_json = {}
                raw_response = response_str
                
                if not response_str:
                    response_str = "{}"
                
                # Try to find JSON in markdown blocks
                if "```json" in str(response_str):
                    response_str = response_str.split("```json")[1].split("```")[0]
                elif "```" in str(response_str):
                    response_str = response_str.split("```")[1].split("```")[0]
                else:
                    # If no backticks, try to find the first '{' and last '}'
                    start = response_str.find("{")
                    end = response_str.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        response_str = response_str[start:end+1]
                    
                try:
                    candidate_json = json.loads(response_str)
                except (json.JSONDecodeError, TypeError):
                    print(f"    Failed to parse JSON from {model}. Response was: {str(raw_response)[:100]}...")
                    candidate_json = {}

                # Append result to list
                all_results.append({
                    "doc_name": doc_name,
                    "model": f"{model} ({itype})",
                    "input_type": itype,
                    "iteration": i,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_json": candidate_json,
                    "elapsed": round(elapsed, 2)
                })

    # Save all results to one file
    output_filename = OUTPUT_DIR / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nInference complete. All results saved to: {output_filename}")

if __name__ == "__main__":
    main()
