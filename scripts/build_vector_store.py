import os
import json
import requests
from pathlib import Path

# --- Configuration ---
OLLAMA_API_URL = "http://192.168.0.15:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text" 
BASE_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = BASE_DIR / "data/examples"
VECTOR_STORE_PATH = BASE_DIR / "data/rag/vector_store.json"

def get_embedding(text):
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def build_store():
    store = []
    
    # Check if examples dir exists
    if not EXAMPLES_DIR.exists():
        print(f"Error: {EXAMPLES_DIR} does not exist.")
        return

    # Find pairs of input/output
    input_files = sorted(EXAMPLES_DIR.glob("*_input.txt"))
    
    print(f"Found {len(input_files)} example inputs.")

    for input_file in input_files:
        base_name = input_file.name.replace("_input.txt", "")
        output_file = EXAMPLES_DIR / f"{base_name}_output.json"
        
        if not output_file.exists():
            print(f"Warning: No output file for {input_file.name}, skipping.")
            continue
            
        print(f"Processing {base_name}...")
        
        with open(input_file, "r", encoding="utf-8") as f:
            text_content = f.read()
            
        with open(output_file, "r", encoding="utf-8") as f:
            json_content = f.read() # Keep as string for prompt injection
            
        # Generate vector
        vector = get_embedding(text_content)
        
        if vector:
            # Determine type from filename for now (simple heuristic)
            doc_type = "unknown"
            if "passport" in base_name.lower():
                doc_type = "passport"
            elif "snils" in base_name.lower():
                doc_type = "snils"
            elif "driver" in base_name.lower() or "права" in base_name.lower():
                doc_type = "driver_license"
            elif "birth" in base_name.lower() or "свидетельство" in base_name.lower():
                doc_type = "birth_certificate"

            store.append({
                "id": base_name,
                "doc_type": doc_type,
                "vector": vector,
                "example_input": text_content,
                "example_output": json_content
            })
            print(f"  -> Added to store.")
        else:
            print(f"  -> Failed to generate vector.")

    # Save store
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
        
    print(f"\nVector store saved to {VECTOR_STORE_PATH} with {len(store)} entries.")

if __name__ == "__main__":
    build_store()
