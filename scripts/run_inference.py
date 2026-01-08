import os
import json
import requests
import time
import logging
from pathlib import Path
from datetime import datetime

# --- Configuration ---
OLLAMA_API_URL = "http://192.168.0.15:11434/api/generate"
BASE_DIR = Path(__file__).parent.parent
OCR_DIR = BASE_DIR / "data/ocr"
REFS_DIR = BASE_DIR / "data/references"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "tests/model_outputs"
LOG_DIR = BASE_DIR / "logs"

ITERATIONS = 3
MODELS = [
    "gemma3:1b",
    "qwen3:1.7b",
    "ministral-3:3b",
    "gemma3:4b",
    "ministral-3:8b",
    "qwen3:8b",
]

# --- Logging Setup ---
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "inference_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


def preload_model(model_name):
    print(f"  Loading model {model_name} into memory...")
    try:
        # Send a dummy request to trigger model loading
        requests.post(
            OLLAMA_API_URL, json={"model": model_name, "prompt": ""}, timeout=600
        )
    except Exception as e:
        msg = f"Failed to preload model {model_name}: {e}"
        print(f"  Warning: {msg}")
        logger.error(msg)


def get_model_response(model_name, prompt, retries=3, delay=5):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": 4096,  # Limiting context to avoid OOM
            "temperature": 0.1,
        },
    }
    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response")
        except requests.exceptions.ConnectionError:
            msg = f"Connection error with Ollama on {model_name} (Attempt {attempt + 1}). Server might have crashed."
            print(f"    {msg}")
            logger.critical(msg)
            # If server crashed, retrying immediately won't help unless it restarts auto.
            # We wait a bit longer.
            time.sleep(10)
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            logger.error(f"Failed to get response from {model_name}: {e}")
            return f"Error: {str(e)}"
    return "Error: Maximum retries exceeded."


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

    # 1. Сбор задач (Task Collection)
    tasks = []
    for ref_file in REFS_DIR.glob("*-reference.json"):
        original_name = ref_file.name.replace("-reference.json", "")

        with open(ref_file, "r", encoding="utf-8") as f:
            reference_json = json.load(f)

        # Select template based on filename
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
            print(
                f"Warning: No template found for {original_name}, using reference as schema hint."
            )
            schema_hint = json.dumps(reference_json, indent=2, ensure_ascii=False)

        # Task for Clean Text ONLY (Removed XML handling as requested)
        clean_file = OCR_DIR / f"{original_name}-clean"
        if clean_file.exists():
            with open(clean_file, "r", encoding="utf-8") as f:
                tasks.append(
                    {
                        "doc_name": original_name,
                        "input_text": f.read(),
                        "input_type": "clean",
                        "schema_hint": schema_hint,
                    }
                )
        else:
            print(f"Warning: Clean file not found for {ref_file.name}")

    print(f"Found {len(tasks)} tasks (docs * types).")

    # Define output filename once at the start
    output_filename = (
        OUTPUT_DIR
        / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    print(f"Results will be saved incrementally to: {output_filename}")

    all_results = []

    # 2. Прогон по моделям
    for model in MODELS:
        print(f"\n--- Testing Model: {model} ---")
        preload_model(model)

        for task in tasks:
            doc_name = task["doc_name"]
            itype = task["input_type"]

            for i in range(1, ITERATIONS + 1):
                print(f"  Processing {doc_name} [{itype}] (run {i}/{ITERATIONS})...")

                context_desc = "OCR text"
                prompt = f"""
You are a helpful assistant that parses OCR data into JSON.
Here is the {context_desc} from a document:
---
{task["input_text"]}
---

Please extract data and return ONLY a valid JSON object matching this structure:
{task["schema_hint"]}

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
                        response_str = response_str[start : end + 1]

                try:
                    candidate_json = json.loads(response_str)
                except (json.JSONDecodeError, TypeError):
                    error_msg = f"Failed to parse JSON from {model} for {doc_name}. Response start: {str(raw_response)[:200]}"
                    print(f"    {error_msg}")
                    logger.error(f"{error_msg}\nFull Response:\n{raw_response}")
                    candidate_json = {}

                # Append result to list
                all_results.append(
                    {
                        "doc_name": doc_name,
                        "model": model,  # Removed "input_type" suffix since we only use clean now
                        "input_type": itype,
                        "iteration": i,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parsed_json": candidate_json,
                        "elapsed": round(elapsed, 2),
                    }
                )

                # Incremental save
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nInference complete. All results saved to: {output_filename}")
    print(f"Errors logged to: {LOG_DIR / 'inference_errors.log'}")


if __name__ == "__main__":
    main()
