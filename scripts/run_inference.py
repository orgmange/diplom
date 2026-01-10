import json
import os
import requests
import re
import argparse
from datetime import datetime
from difflib import SequenceMatcher
import numpy as np

# --- Configuration ---
OLLAMA_HOST = "http://192.168.0.15:11434"
DATA_DIR = "data"
OCR_DIR = os.path.join(DATA_DIR, "ocr")
REFERENCES_DIR = os.path.join(DATA_DIR, "references")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "rag", "vector_store.json")
OUTPUT_DIR = "tests/model_outputs"
LOG_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# List of models to test
MODELS = [
    "gemma3:1b",
    "qwen2.5:1.5b",
    "ministral-3:3b",
    "gemma3:4b",
    "ministral:8b",
    "qwen2.5:7b"
]

def get_embedding(text):
    """
    Get the embedding for a given text using Ollama's nomic-embed-text model.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def load_vector_store():
    """
    Loads the vector store from disk.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. RAG will be disabled.")
        return None
    
    with open(VECTOR_STORE_PATH, 'r', encoding='utf-8') as f:
        store = json.load(f)
    
    # Convert embeddings to numpy arrays for faster processing
    for item in store:
        item['vector'] = np.array(item['vector'])
        
    return store

def find_most_similar(query_embedding, vector_store):
    """
    Finds the most similar example in the vector store using cosine similarity.
    """
    if not vector_store or query_embedding is None:
        return None
    
    query_vec = np.array(query_embedding)
    best_match = None
    max_similarity = -1
    
    # Normalize query vector
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return None
        
    for item in vector_store:
        doc_vec = item['vector']
        doc_norm = np.linalg.norm(doc_vec)
        
        if doc_norm == 0:
            continue
            
        similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = item
            
    return best_match

def load_reference(image_filename):
    """
    Loads the ground truth JSON for a given image file.
    Assumes reference files are named <image_filename>-reference.json
    """
    ref_path = os.path.join(REFERENCES_DIR, f"{image_filename}-reference.json")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def clean_json_string(json_string):
    """
    Heuristic to extract valid JSON from a potentially messy LLM response.
    """
    # Try to find the first '{' and the last '}'
    start_idx = json_string.find('{')
    end_idx = json_string.rfind('}')

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        potential_json = json_string[start_idx : end_idx + 1]
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass # Continue if simple extraction fails

    # Cleanup markdown code blocks if present
    json_string = re.sub(r'^```json', '', json_string, flags=re.MULTILINE)
    json_string = re.sub(r'^```', '', json_string, flags=re.MULTILINE)
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None

def calculate_accuracy(pred_json, true_json):
    """
    Calculates a simple accuracy metric based on matching string values.
    Flattens nested JSONs for comparison.
    """
    def flatten(y):
        out = {}
        def flatten_recursive(x, name=''):
            if type(x) is dict:
                for a in x:
                    flatten_recursive(x[a], name + a + '_')
            elif type(x) is list:
                pass # Ignoring lists for simplicity in this version
            else:
                out[name[:-1]] = str(x).lower().strip()
        flatten_recursive(y)
        return out

    flat_pred = flatten(pred_json)
    flat_true = flatten(true_json)
    
    match_count = 0
    total_fields = len(flat_true)
    
    if total_fields == 0:
        return 0.0

    for key, val in flat_true.items():
        if key in flat_pred:
            # Simple string similarity
            if SequenceMatcher(None, val, flat_pred[key]).ratio() > 0.8: # Fuzzy match threshold
                match_count += 1
            
    return (match_count / total_fields) * 100

def run_inference():
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f"inference_results_{timestamp}.json")
    
    all_results = []
    
    # Get list of OCR files
    ocr_files = [f for f in os.listdir(OCR_DIR) if f.endswith("-clean")]
    
    if not ocr_files:
        print("No OCR text files found in data/ocr/")
        return

    print(f"Found {len(ocr_files)} OCR files to process.")

    for i, ocr_file in enumerate(ocr_files):
        image_filename = ocr_file.replace("-clean", "") # Assuming OCR text file matches image filename + -clean
        ocr_path = os.path.join(OCR_DIR, ocr_file)
        
        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()
            
        reference_data = load_reference(image_filename)
        if not reference_data:
            print(f"Skipping {image_filename}: No reference data found.")
            continue

        print(f"Processing {image_filename} ({i+1}/{len(ocr_files)})...")
        
        # --- RAG: Get Embedding and Retrieve Context ---
        rag_context = ""
        if vector_store:
            embedding = get_embedding(ocr_text)
            similar_doc = find_most_similar(embedding, vector_store)
            
            if similar_doc:
                print(f"  RAG: Found similar example: {similar_doc['doc_type']}")
                rag_context = (
                    "Here is an example of how to extract data from a similar document.\n"
                    "EXAMPLE INPUT:\n"
                    f"{similar_doc['example_input']}\n\n"
                    "EXAMPLE OUTPUT:\n"
                    f"{similar_doc['example_output']}\n\n"
                    "Now, extract data from the following text in the same JSON format:\n"
                )
            else:
                 rag_context = "Extract the following data into a structured JSON format.\n"
        else:
             rag_context = "Extract the following data into a structured JSON format.\n"


        for model in MODELS:
            print(f"  Running model: {model}")
            
            prompt = (
                f"You are a helpful AI assistant that extracts structured data from OCR text of Russian identity documents.\n"
                f"{rag_context}"
                f"INPUT TEXT:{ocr_text}\n\n"
                f"Return ONLY the valid JSON object. Do not include any other text."
            )
            
            start_time = datetime.now()
            try:
                response = requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1 # Low temperature for more deterministic output
                        }
                    },
                    timeout=120 # Increased timeout for larger models
                )
                response.raise_for_status()
                result_json = response.json()
                raw_response = result_json["response"]
                duration = (datetime.now() - start_time).total_seconds()
                
                parsed_json = clean_json_string(raw_response)
                
                if parsed_json:
                    accuracy = calculate_accuracy(parsed_json, reference_data)
                    status = "success"
                else:
                    accuracy = 0.0
                    status = "json_parse_error"
                    with open(os.path.join(LOG_DIR, "inference_errors.log"), "a", encoding="utf-8") as log:
                        log.write(f"[{datetime.now()}] {model} - {image_filename} - JSON Parse Error\nRaw Response: {raw_response}\n\n")

                result_entry = {
                    "model": model,
                    "document": image_filename,
                    "duration_seconds": duration,
                    "accuracy": accuracy,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    "predicted_data": parsed_json
                }
                
                all_results.append(result_entry)
                
                # Incremental Save
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"  Error with model {model}: {e}")
                with open(os.path.join(LOG_DIR, "inference_errors.log"), "a", encoding="utf-8") as log:
                    log.write(f"[{datetime.now()}] {model} - {image_filename} - Exception: {str(e)}\n")

    print(f"\nInference complete. Results saved to {results_file}")

if __name__ == "__main__":
    run_inference()