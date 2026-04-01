import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "Diplom OCR & RAG System"
    API_V1_STR: str = "/api/v1"
    
    # OCR Settings
    OCR_API_KEY: str = os.getenv("OCR_API_KEY", "")
    OCR_BASE_URL: str = "https://ocrbot.ru/api/v1"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    DOCS_DIR: Path = DATA_DIR / "docs"
    BENCHMARK_REF_DIR: Path = DATA_DIR / "references"
    
    # Qdrant Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME: str = "documents"
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBED_BASE_URL: str = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11435")
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    DEFAULT_MODEL: str = "qwen3.5:9b"

settings = Settings()
