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
                ex_text = str(ex.get('cleaned_text', ''))[:4000]
                prompt += f"OCR INPUT:\n{ex_text}\n\n"
                prompt += f"JSON OUTPUT:\n{ex.get('json_output')}\n\n"
            prompt += "--- END OF EXAMPLES ---\n\n"

        prompt += "Now, process the following OCR text and return ONLY the JSON object. "
        prompt += "If a field is missing, use an empty string or null. Ensure the output is valid JSON.\n\n"
        prompt += f"### TARGET OCR TEXT:\n{target_text}\n\n"
        prompt += "### FINAL JSON OUTPUT:"
        
        return prompt

    def _get_schema_for_type(self, doc_type: Optional[str]) -> Optional[Dict[str, Any]]:
        """Загружает шаблон и преобразует его в упрощенную JSON Schema для Ollama."""
        if not doc_type:
            return None
            
        template_map = {
            "passport": "passport_ru.json",
            "driver_license": "driver_license_ru.json",
            "snils": "snils.json",
            "birth_certificate": "birth_certificate_ru.json"
        }
        
        template_file = template_map.get(doc_type)
        if not template_file:
            return None
            
        try:
            template_path = settings.BASE_DIR / "templates" / template_file
            if not template_path.exists():
                return None
                
            with open(template_path, "r", encoding="utf-8") as f:
                template_data = json.load(f)
                
            # Генерируем простую схему, где все поля - строки
            properties = {}
            for key, val in template_data.items():
                if isinstance(val, dict):
                    nested_props = {k: {"type": "string"} for k in val.keys()}
                    properties[key] = {
                        "type": "object",
                        "properties": nested_props
                    }
                else:
                    properties[key] = {"type": "string"}
                    
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys())
            }
        except Exception as e:
            logger.error(f"Error generating schema for {doc_type}: {e}")
            return None

    def structure(
        self,
        raw_text: str,
        cleaned_text: str,
        model_name: str,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Основной метод: ищет похожий пример, строит промпт и вызывает LLM.
        """
        # 1. Поиск наиболее похожего примера по сырому тексту
        logger.info(f"Searching for examples using model: {embedding_model or 'default'}")
        
        # Берем 3 примера, чтобы выбрать лучший по doc_type или отсечь шум
        all_results = self.vector_service.search(
            raw_text,
            limit=3,
            only_examples=True,
            embedding_model=embedding_model,
        )
        
        # Фильтруем по порогу сходства
        threshold = 0.4
        examples = [r for r in all_results if r.get('score', 0) >= threshold]
        
        doc_type = None
        if examples:
            # Берем тип из самого релевантного примера
            best_match = examples[0]
            doc_type = best_match.get('doc_type')
            logger.info(f"RAG Match: {best_match.get('filename')} (type: {doc_type}, score: {best_match.get('score'):.4f})")
            
            # Оставляем только 1 лучший пример для Few-Shot промпта, 
            # чтобы не раздувать контекст и не путать модель разными типами
            examples = [best_match]
        else:
            if all_results:
                logger.warning(f"Best RAG match below threshold: {all_results[0].get('filename')} (score: {all_results[0].get('score'):.4f})")
            else:
                logger.warning("No examples found in vector store.")
            
            # Fallback: определяем тип по тексту
            doc_type = self.vector_service._detect_doc_type(cleaned_text[:200])
            logger.info(f"Fallback detection: {doc_type}")

        # 2. Формирование промпта и схемы
        truncated_cleaned = cleaned_text[:8000]
        prompt = self.build_prompt(truncated_cleaned, examples)
        schema = self._get_schema_for_type(doc_type)
        
        if schema:
            logger.info(f"Using structured output schema for type: {doc_type}")
        else:
            logger.warning(f"No schema found for type: {doc_type}. Falling back to generic JSON.")
        
        if schema:
            logger.info(f"Using structured output schema for type: {doc_type}")
        else:
            logger.warning(f"No schema found for type: {doc_type}. Falling back to generic JSON.")
        
        # 3. Вызов Ollama
        logger.info(f"Calling Ollama model '{model_name}' for structuring (structured output={bool(schema)})...")
        logger.debug(f"PROMPT SENT TO LLM:\n{prompt}")
        
        result_str = ""
        try:
            response = self.ollama_client.generate(
                model=model_name,
                prompt=prompt,
                format=schema or "json",
                options={"temperature": 0.1}
            )
            
            result_str = response.get('response', '').strip()
            logger.debug(f"RAW RESPONSE FROM LLM:\n{result_str}")
            
            if not result_str:
                # В случае пустого ответа логируем промпт в ERROR для дебага
                logger.error(f"MODEL RETURNED EMPTY RESPONSE. Prompt snippet: {prompt[:500]}...")
                raise ValueError("Empty response from model")

            return json.loads(result_str)
                
        except Exception as e:
            logger.error(f"Failed to structure with model {model_name}: {e}")
            logger.error(f"RAW RESPONSE AT MOMENT OF ERROR: {result_str}")
            
            # Fallback: пробуем найти JSON если схема не сработала или произошла ошибка
            if result_str:
                try:
                    import re
                    json_match = re.search(r'(\{.*\})', result_str, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                except:
                    pass

            return {
                "error": str(e),
                "model": model_name,
                "doc_type": doc_type,
                "raw_response_snippet": result_str[:100] if result_str else ""
            }
