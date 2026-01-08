# Devlog: OCR Parsing Module with LLM & RAG

## Overview
Development of a module for parsing OCR data using LLMs with RAG, running models via Ollama.

## Tested Models
- gemma3:1b
- gemma3:4b
- qwen3:8b
- ~~qwen3:4b~~ (FAILED - Does not launch/run)
- qwen3:1.7b
- ministral-3:8b (Potential candidate for best performance)
- ministral-3:3b

## Status & Observations
- **Done:** Tested models on 5 documents (pre-parsed by OCR bot and cleaned of XML).
- **Issue:** `qwen3:4b` failed to execute completely. Other models occasionally hallucinate structure.
- **Key Insight:** Data cleaning is strictly necessary.
- **Hypothesis:** `ministral-3:8b` seems promising.
- **Next Step:** Implement RAG (Few-Shot Learning) to improve stability.

## RAG Implementation Plan (Планирование RAG: Vector-based Routing)

### Концепция: "Semantic Router via Embeddings"
Мы используем векторные представления (embeddings) для определения типа документа. Это позволит игнорировать ошибки OCR, так как сравнение идет по смыслу, а не по точному совпадению символов.

### 1. Подготовка Базы (Vector Store)
- **Модель эмбеддингов:** Использовать Ollama (например, `nomic-embed-text` или `all-minilm`).
- **Эталоны:** Для каждого типа документа (Паспорт, СНИЛС, Права, Свидетельство) создаем текстовое описание или берем типичный OCR-текст.
- **Прекомпут:** Заранее вычисляем векторы для эталонов и сохраняем их.

### 2. Пайплайн обработки (The Flow)
1. **Входящий документ:** Читаем грязный OCR-текст.
2. **Embedding Generation:** Отправляем текст в Ollama `/api/embeddings` -> получаем вектор `Vec_Input`.
3. **Similarity Search (k-NN):**
   - Считаем Косинусное сходство (Cosine Similarity) между `Vec_Input` и векторами эталонов.
   - Побеждает эталон с максимальным `score`.
4. **Context Injection:**
   - На основе победителя выбираем соответствующий промпт, схему JSON и One-Shot пример.
5. **Generation:** Запрос к LLM уже с правильным контекстом.

### 3. Техническая реализация
- **Шаг 1:** Выбрать и загрузить embedding-модель в Ollama.
- **Шаг 2:** Создать скрипт `scripts/build_vector_store.py`, который пробегает по `templates/` или примерам и создает файл `data/vector_store.json`.
- **Шаг 3:** В `run_inference.py` добавить вызов `/api/embeddings` и функцию расчета косинусного расстояния.

## Task List

### Completed
- [x] Initial setup of Ollama environment.
- [x] Run initial tests on 5 OCR-processed documents.
- [x] Verify the necessity of data cleaning.
- [x] Identify failing models (`qwen3:4b`).

### To Do / In Progress
- [ ] **Data Prep:** Create manually verified example pairs (Input/JSON) for Passport, DLC, SNILS, Birth Cert.
- [ ] **Code:** Modify `run_inference.py` to support dynamic example injection.
- [ ] **Test:** Run `ministral-3:8b` and `gemma3:4b` with RAG enabled.
- [ ] **Evaluate:** Compare accuracy against previous runs.