import json
import logging
import ollama
from typing import List, Dict, Any, Optional
from app.services.vector_service import VectorService
from app.core.config import settings

logger = logging.getLogger("diplom")

class StructuringService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self._ollama_client = None

    @property
    def ollama_client(self):
        if not self._ollama_client:
            # Асинхронный клиент с таймаутом 1 минута
            self._ollama_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL, timeout=60.0)
        return self._ollama_client

    async def get_available_models(self) -> List[str]:
        try:
            response = await self.ollama_client.list()
            return [m.model for m in response.models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise e

    def _get_template_for_type(self, doc_type: str) -> str:
        template_map = {
            "passport": "passport_ru.json",
            "driver_license": "driver_license_ru.json",
            "snils": "snils.json",
            "birth_certificate": "birth_certificate_ru.json"
        }
        filename = template_map.get(doc_type)
        if not filename: return "{}"
        try:
            path = settings.BASE_DIR / "templates" / filename
            return path.read_text(encoding="utf-8")
        except: return "{}"

    async def structure(
        self,
        raw_text: str,
        cleaned_text: str,
        model_name: str,
        embedding_model: Optional[str] = None,
        expected_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Стабильный асинхронный метод с большим контекстом (16к) и полными примерами RAG.
        """
        # 1. RAG
        all_results = self.vector_service.search(
            cleaned_text, limit=1, only_examples=True, embedding_model=embedding_model
        )
        best_match = all_results[0] if all_results and all_results[0].get('score', 0) > 0.4 else None
        doc_type = best_match.get('doc_type') if best_match else (expected_type or self.vector_service._detect_doc_type(cleaned_text[:200]))
        
        # 2. Подготовка промпта
        template_json = self._get_template_for_type(doc_type)
        system_msg = (
            "You are a professional data extraction system. "
            "Extract data from OCR text according to the provided JSON schema. "
            "Return ONLY raw JSON object."
        )
        
        user_content = f"### JSON SCHEMA TO FOLLOW:\n{template_json}\n\n"
        if best_match:
            # Используем ПОЛНЫЙ текст примера для лучшего качества Few-Shot
            user_content += f"### REFERENCE EXAMPLE:\nINPUT: {best_match.get('cleaned_text')}\nOUTPUT: {best_match.get('json_output')}\n\n"
        
        # Передаем полный очищенный текст OCR
        user_content += f"### TARGET OCR TEXT TO PROCESS:\n{cleaned_text}\n\n"
        user_content += "### FINAL RESULT (JSON ONLY):"

        messages = [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_content},
        ]
        
        logger.debug(f"MESSAGES SENT TO LLM (Type: {doc_type}, Chars: {len(user_content)})")
        
        # Определяем модель (по умолчанию qwen3:8b если передано llama3)
        actual_model = model_name
        if "llama3" in model_name.lower():
            actual_model = "qwen3:8b"
            
        max_retries = 3
        last_error = ""
        import asyncio

        for attempt in range(max_retries):
            try:
                response = await self.ollama_client.chat(
                    model=actual_model, 
                    messages=messages,
                    format='json',
                    options={
                        "temperature": 0.0
                    }
                )
                
                result_str = response.message.content.strip()
                if not result_str:
                    raise ValueError("Ollama returned empty response")

                logger.debug(f"RAW LLM RESPONSE (Len: {len(result_str)})")
                
                return {
                    "result": json.loads(result_str),
                    "doc_type": doc_type,
                    "model": actual_model
                }
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Ollama attempt {attempt + 1}/{max_retries} failed for {actual_model}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                
        logger.error(f"Ollama failure after {max_retries} attempts: {last_error}")
        return {
            "result": {"error": f"Ollama failure after {max_retries} attempts: {last_error}"},
            "doc_type": doc_type,
            "model": actual_model
        }

