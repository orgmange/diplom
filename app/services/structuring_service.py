import json
import asyncio
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
            # Асинхронный клиент с таймаутом 1.5 минуты
            self._ollama_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL, timeout=90.0)
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
            "birth_certificate": "birth_certificate_ru.json",
            "diplom_bakalavra": "diplom.json",
            "dogovor_kupli_prodazhi_kv": "dogovor_kupli_kv.json",
            "dogovor_prodagi_machini": "dogovor_kupli.json",
            "dogovor_arendi_kv": "renal.json",
            "inn": "inn.json",
            "kvitancia": "kvit.json",
            "zagran_passport": "zagran.json"
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
        on_chunk: Optional[Any] = None,
        temperature: float = 0.0,
        num_ctx: int = 16383,
        timeout: int = 60,
        structured_output: bool = True
    ) -> Dict[str, Any]:
        """
        Стабильный асинхронный метод с настраиваемыми параметрами.
        """
        # 1. RAG (Поиск типа документа и примера для LLM)
        # Ищем по единой базе примеров
        all_results = self.vector_service.search(
            cleaned_text, limit=1, embedding_model=embedding_model
        )
        
        best_match = all_results[0] if all_results and all_results[0].get('score', 0) > 0.4 else None
        
        # Тип документа берем из метаданных найденного примера
        doc_type = best_match.get('doc_type') if best_match else (expected_type or "unknown")
        
        # 2. Подготовка промпта
        template_json = self._get_template_for_type(doc_type)
        system_msg = (
            "You are a professional data extraction system. "
            "Extract data from OCR text according to the provided JSON schema. "
            "Return ONLY raw JSON object."
        )
        
        user_content = f"### JSON SCHEMA TO FOLLOW:\n{template_json}\n\n"
        if best_match:
            # Используем текст и JSON найденного примера для Few-Shot обучения
            user_content += f"### REFERENCE EXAMPLE:\nINPUT: {best_match.get('text')}\nOUTPUT: {best_match.get('json_output')}\n\n"
        
        # Передаем очищенный текст OCR
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
            actual_model = "qwen3.5:9b"
            
        max_retries = 3
        last_error = ""


        for attempt in range(max_retries):
            try:
                # Включаем стриминг для детектирования активности модели
                logger.debug(f"Starting streamed chat with {actual_model} (Attempt {attempt+1})")
                full_content = ""
                first_chunk = True
                
                # Используем вспомогательную переменную для отслеживания таймаута между чанками
                # Для первого чанка даем настраиваемый таймаут (на случай загрузки модели),
                # для последующих - по 5 секунд ожидания.
                
                stream = await self.ollama_client.chat(
                    model=actual_model, 
                    messages=messages,
                    format='json' if structured_output else None,
                    options={
                        "temperature": temperature,
                        "num_ctx": num_ctx
                    },
                    stream=True,
                    think=False
                )

                while True:
                    try:
                        # Таймаут на ожидание следующего чанка
                        wait_time = float(timeout) if first_chunk else 5.0
                        chunk = await asyncio.wait_for(stream.__anext__(), timeout=wait_time)
                        
                        if first_chunk:
                            logger.debug(f"Ollama started generating content for {actual_model}")
                            first_chunk = False
                        
                        if chunk.message.content:
                            full_content += chunk.message.content
                            if on_chunk:
                                await on_chunk(chunk.message.content)
                                
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.error(f"Ollama stream timed out after {wait_time}s of inactivity")
                        raise ValueError(f"Ollama inactivity timeout ({wait_time}s)")

                if not full_content.strip():
                    raise ValueError("Ollama returned empty response")

                logger.debug(f"RAW LLM RESPONSE (Len: {len(full_content)})")
                
                return {
                    "result": json.loads(full_content),
                    "doc_type": doc_type,
                    "model": actual_model,
                    "prompt_size": sum(len(m['content']) for m in messages)
                }
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Ollama attempt {attempt + 1}/{max_retries} failed for {actual_model}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                
        logger.error(f"Ollama failure after {max_retries} attempts for model {actual_model}.")
        logger.error(f"LAST PROMPT:\n{json.dumps(messages, ensure_ascii=False, indent=2)}")
        if full_content:
            logger.error(f"LAST PARTIAL RESPONSE:\n{full_content}")
            
        return {
            "result": {"error": f"Ollama failure after {max_retries} attempts: {last_error}"},
            "doc_type": doc_type,
            "model": actual_model,
            "prompt_size": sum(len(m['content']) for m in messages),
            "raw_response": full_content,
            "prompt": messages
        }

