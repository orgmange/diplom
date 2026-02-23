import json
import logging
import ollama
from typing import List, Dict, Any, Optional
from app.services.vector_service import VectorService
from app.core.config import settings

logger = logging.getLogger("diplom")

class StructuringService:
    """
    Сервис для структурирования данных при помощи LLM и RAG (few-shot).
    """
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self._ollama_client = None

    @property
    def ollama_client(self):
        if not self._ollama_client:
            self._ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        return self._ollama_client

    def get_available_models(self) -> List[str]:
        """Возвращает список доступных моделей Ollama. Пробрасывает ошибку при неудаче."""
        try:
            # В ollama-python 0.6.1 list() возвращает ListResponse с атрибутом models
            response = self.ollama_client.list()
            return [m.model for m in response.models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise e

    def build_prompt(self, target_text: str, examples: List[Dict]) -> str:
        """Формирует промпт с примерами (few-shot) для LLM."""
        prompt = "You are an expert in extracting structured data from OCR text. "
        prompt += "Your task is to convert the OCR text into a VALID JSON object.\n\n"
        
        if examples:
            prompt += "### EXAMPLES OF EXTRACTION:\n\n"
            for i, ex in enumerate(examples):
                prompt += f"--- EXAMPLE {i+1} ---\n"
                prompt += f"OCR INPUT:\n{ex.get('cleaned_text')}\n\n"
                prompt += f"JSON OUTPUT:\n{ex.get('json_output')}\n\n"
            prompt += "--- END OF EXAMPLES ---\n\n"

        prompt += "Now, process the following OCR text and return ONLY the JSON object. "
        prompt += "If a field is missing, use an empty string or null. Ensure the output is valid JSON.\n\n"
        prompt += f"### TARGET OCR TEXT:\n{target_text}\n\n"
        prompt += "### FINAL JSON OUTPUT:"
        
        return prompt

    def structure(self, raw_text: str, cleaned_text: str, model_name: str) -> Dict[str, Any]:
        """
        Основной метод: ищет похожий пример, строит промпт и вызывает LLM.
        """
        # 1. Поиск наиболее похожего примера по сырому тексту
        logger.info(f"Searching for examples for raw text (len={len(raw_text)})")
        examples = self.vector_service.search(raw_text, limit=1, only_examples=True)
        
        if examples:
            logger.info(f"Found best example: {examples[0].get('filename')} (score: {examples[0].get('score'):.4f})")
        else:
            logger.warning("No examples found in vector store.")

        # 2. Формирование промпта
        prompt = self.build_prompt(cleaned_text, examples)
        
        # 3. Вызов Ollama с JSON форматом
        logger.info(f"Calling Ollama model '{model_name}' for structuring...")
        try:
            response = self.ollama_client.generate(
                model=model_name,
                prompt=prompt,
                format="json",
                options={"temperature": 0.1} # Низкая температура для стабильности
            )
            
            result_str = response.get('response', '{}')
            return json.loads(result_str)
        except Exception as e:
            logger.error(f"Failed to structure with model {model_name}: {e}")
            return {
                "error": str(e),
                "model": model_name,
                "prompt_used": prompt[:200] + "..."
            }
