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
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen3.5:4b")
    
    # Редиректы моделей (если в названии есть ключ, заменяем на значение)
    MODEL_REPLACEMENTS: dict = {
        "llama3": "qwen3.5:9b"
    }

    def get_actual_model(self, model_name: str) -> str:
        if not model_name:
            return self.DEFAULT_MODEL
        
        m_lower = model_name.lower()
        for key, replacement in self.MODEL_REPLACEMENTS.items():
            if key in m_lower:
                return replacement
        return model_name

    # Database Settings
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "diplom_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "diplom_password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "diplom_db")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
