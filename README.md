# Diplom OCR & RAG System

Интеллектуальная система для распознавания и структурирования данных из сканов российских документов (паспорт, СНИЛС, водительское удостоверение, свидетельство о рождении). Использует OCR для извлечения текста, RAG (Retrieval-Augmented Generation) для определения типа документа и Few-Shot примеров, и LLM (Ollama) для структурирования данных в JSON.

---

## Содержание

- [Архитектура системы](#архитектура-системы)
- [Как работает система](#как-работает-система)
- [Структура проекта](#структура-проекта)
- [Описание файлов](#описание-файлов)
  - [Корневые файлы](#корневые-файлы)
  - [app/ — Бэкенд (FastAPI)](#app--бэкенд-fastapi)
  - [frontend/ — Веб-интерфейс](#frontend--веб-интерфейс)
  - [templates/ — JSON-шаблоны документов](#templates--json-шаблоны-документов)
  - [scripts/ — Утилиты](#scripts--утилиты)
  - [data/ — Данные](#data--данные)
  - [tests/ — Тесты](#tests--тесты)
- [Установка и запуск](#установка-и-запуск)
- [API Endpoints](#api-endpoints)

---

## Архитектура системы

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Изображение │────▶│   OCR API    │────▶│   Cleaner    │────▶│   RAG поиск  │
│  документа   │     │  (ocrbot.ru) │     │ (XML → Text) │     │   (Qdrant)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │   LLM        │
                                                               │  (Ollama)    │
                                                               │  + шаблон    │
                                                               │  + пример    │
                                                               └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │ JSON-результ │
                                                               │ (структурир. │
                                                               │  данные)     │
                                                               └──────────────┘
```

**Технологический стек:**
- **Backend**: Python 3, FastAPI, Uvicorn
- **LLM**: Ollama (локальный сервер, порт 11434 для генерации, порт 11435 для эмбеддингов)
- **Векторная БД**: Qdrant (Docker, порт 6333)
- **OCR**: Внешний API [ocrbot.ru](https://ocrbot.ru)
- **Frontend**: Vanilla HTML/CSS/JS (SPA)

---

## Как работает система

### Полный цикл обработки документа

1. **OCR (Распознавание)** — Изображение документа отправляется на внешний API `ocrbot.ru`. Результат — XML-файл с координатами слов, строк и уровнями уверенности распознавания.

2. **Очистка (Cleaning)** — XML-ответ парсится, извлекаются текстовые строки. Строки сортируются по вертикали (Y-координата), группируются близкие по высоте строки в одну логическую строку, затем сортируются по горизонтали (X). Результат — чистый текст без XML-разметки.

3. **RAG-поиск (Retrieval)** — Очищенный текст превращается в вектор через embedding-модель (Ollama, порт 11435, модель `nomic-embed-text`). Вектор используется для поиска ближайшего примера в Qdrant. При этом определяется **тип документа** (паспорт, СНИЛС и т.д.) из метаданных найденного примера.

4. **Структурирование (Generation)** — Формируется промпт для LLM:
   - Системное сообщение: роль «data extraction system»
   - JSON-шаблон для определённого типа документа (из `templates/`)
   - Few-Shot пример: найденный через RAG ближайший пример (вход + выход)
   - Целевой OCR-текст для обработки
   
   LLM возвращает структурированный JSON с полями документа.

### Система бенчмаркинга

Система поддерживает два типа бенчмарков:

- **Embedding-бенчмарк** — Проверяет, насколько точно embedding-модель находит правильный тип документа по OCR-тексту. Берёт очищенные тексты из `data/docs/*/clean/`, ищет ближайший пример в Qdrant и сравнивает предсказанный тип с ожидаемым.

- **Structuring-бенчмарк** — Полный цикл: OCR → Clean → RAG → LLM. Сравнивает JSON-результат с эталонным файлом из `data/references/`. Вычисляет метрики: Accuracy, Precision, Recall, F1-score, CER (Character Error Rate), Fuzzy Score. Поддерживает запуск по нескольким моделям одновременно с сохранением истории отчётов.

### Внешнее API (для интеграции)

Система предоставляет асинхронное API для внешних сервисов:
- Отправка изображения → получение `task_id` → опрос статуса → получение результата (JSON или XML)
- Два режима: **template** (только OCR, возврат XML) и **recognize** (полный цикл, возврат структурированного JSON)

---

## Структура проекта

```
diplom/
├── app/                          # Бэкенд (FastAPI)
│   ├── main.py                   # Точка входа приложения
│   ├── api/
│   │   ├── endpoints.py          # Внутренние API-эндпоинты
│   │   └── external.py           # Внешнее API (распознавание)
│   ├── core/
│   │   └── config.py             # Конфигурация (пути, ключи, порты)
│   └── services/
│       ├── ocr_service.py        # Взаимодействие с OCR API
│       ├── cleaner_service.py    # Парсинг XML → чистый текст
│       ├── vector_service.py     # Qdrant + Ollama Embeddings
│       ├── structuring_service.py     # RAG + LLM структурирование
│       ├── recognition_service.py     # Асинхронный пайплайн
│       ├── benchmark_service.py       # Embedding-бенчмарк
│       └── structuring_benchmark_service.py  # LLM-бенчмарк
├── frontend/                     # Веб-интерфейс (SPA)
│   ├── index.html                # HTML-разметка
│   ├── app.js                    # Логика UI
│   └── styles.css                # Стили
├── templates/                    # JSON-шаблоны документов
│   ├── passport_ru.json
│   ├── driver_license_ru.json
│   ├── snils.json
│   └── birth_certificate_ru.json
├── scripts/                      # Утилиты и скрипты
│   ├── ocr_fetcher.py            # Standalone OCR-скрипт
│   ├── ocr_cleaner.py            # Standalone очистка XML
│   ├── generate_references.py    # Генерация эталонных JSON
│   ├── check_vectors.py          # Инспекция Qdrant
│   └── debug_search.py           # Отладка поиска
├── data/                         # Данные (документы, примеры, RAG)
│   ├── docs/                     # Документы по типам
│   ├── examples/                 # Few-Shot примеры (вход + выход)
│   ├── references/               # Эталонные JSON для бенчмарков
│   ├── ocr/                      # Исторические OCR-файлы
│   ├── rag/                      # Хранилище Qdrant + state.json
│   ├── benchmark/                # Отчёты бенчмарков
│   └── temp/                     # Временные файлы
├── tests/                        # Юнит-тесты (pytest)
├── cli.py                        # CLI-интерфейс
├── docker-compose.yaml           # Docker для Qdrant
├── requirements.txt              # Python-зависимости
├── .env                          # Переменные окружения (секреты)
├── .env.example                  # Пример .env
├── .gitignore                    # Исключения Git
└── RULES.md                      # Правила разработки
```

---

## Описание файлов

### Корневые файлы

| Файл | Назначение |
|------|------------|
| `cli.py` | CLI-интерфейс для запуска OCR, очистки, индексации и поиска из командной строки. Команды: `ocr`, `clean`, `index`, `search --query "текст"`. |
| `docker-compose.yaml` | Конфигурация Docker для запуска **Qdrant** (векторная БД) на портах 6333/6334. Данные хранятся в `./data/rag`. |
| `requirements.txt` | Список Python-зависимостей: FastAPI, Uvicorn, Ollama, Qdrant Client, RapidFuzz, dicttoxml и другие. |
| `.env` / `.env.example` | Переменные окружения: `OCR_API_KEY` (ключ для ocrbot.ru), `OLLAMA_BASE_URL`, `OLLAMA_EMBED_BASE_URL`, `QDRANT_HOST`, `QDRANT_PORT`. |
| `.gitignore` | Исключает `__pycache__/`, `data/images/`, `.gemini/`, `data/rag/` из Git. |
| `RULES.md` | Правила разработки: без комментариев в коде, docstrings на русском, атомарные коммиты с префиксом `step:`, обязательные тесты, типизация. |

---

### app/ — Бэкенд (FastAPI)

#### `app/main.py`
Точка входа FastAPI-приложения. Выполняет:
- Настройку логирования (в файл `app.log` и в консоль)
- Создание экземпляра `FastAPI` с названием проекта и путём к OpenAPI
- Подключение двух роутеров: `api_router` (внутренние эндпоинты) и `external_router` (внешнее API)
- Раздачу статического фронтенда из директории `frontend/`
- Маршрутизацию корневого пути `/` на `index.html`

#### `app/core/config.py`
Централизованная конфигурация системы (класс `Settings`):
- **OCR**: API-ключ и URL для `ocrbot.ru`
- **Пути**: `BASE_DIR`, `DATA_DIR`, `OCR_DIR`, `DOCS_DIR`, `BENCHMARK_REF_DIR`
- **Qdrant**: хост, порт (6333), имя коллекции (`documents`)
- **Ollama**: URL для генерации (порт 11434), URL для эмбеддингов (порт 11435), модель эмбеддинга (`nomic-embed-text:latest`)

Загружает `.env` файл при инициализации.

#### `app/api/endpoints.py`
Определяет все внутренние API-эндпоинты (29 маршрутов):

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/ocr/scan` | POST | Сканирует `data/docs/*/image/` и запускает OCR для новых изображений |
| `/ocr/clean` | POST | Очищает XML из `data/docs/*/xml/` → `data/docs/*/clean/` |
| `/rag/models` | GET | Список доступных LLM-моделей в Ollama |
| `/rag/benchmark/models` | GET | Список embedding-моделей для бенчмарков |
| `/rag/index_examples` | POST | Индексирует Few-Shot примеры из `data/examples/` в Qdrant |
| `/rag/index` | POST | Алиас для индексации примеров |
| `/rag/reindex` | POST | Полная переиндексация (сброс + переиндексация примеров) |
| `/rag/search` | POST | Семантический поиск по Qdrant с фильтрацией по типу документа |
| `/rag/structure` | POST | Структурирование выбранного файла через LLM |
| `/rag/benchmark/run` | POST | Запуск embedding-бенчмарка |
| `/rag/benchmark/structuring/run` | POST | Запуск LLM-бенчмарка для одной модели |
| `/rag/benchmark/structuring/run-multi` | POST | Запуск LLM-бенчмарка для нескольких моделей |
| `/rag/benchmark/structuring/progress` | GET | Текущий прогресс LLM-бенчмарка |
| `/rag/benchmark/structuring/reports` | GET/DELETE | CRUD для отчётов бенчмарков |
| `/rag/benchmark/cancel` | POST | Отмена текущих бенчмарков |
| `/rag/benchmark/structuring/skip` | POST | Пропуск текущей модели в мульти-бенчмарке |
| `/status` | GET | Статус системы (количество файлов, векторов, текущая модель) |

Также определяет Pydantic-модели для всех запросов и ответов (SearchQuery, StructureRequest, BenchmarkRunResponse и др.).

#### `app/api/external.py`
Внешнее API для интеграции с другими сервисами. Предоставляет асинхронный интерфейс:

| Эндпоинт | Метод | Описание |
|----------|-------|----------|
| `/templates` | POST | Отправка base64-изображения на OCR → возврат `task_id` |
| `/templates/upload` | POST | То же, но принимает файл (для Swagger) |
| `/templates/{task_id}/status` | GET | Статус задачи OCR |
| `/templates/{task_id}/result` | GET | Результат OCR (XML-строка) |
| `/recognize` | POST | Полный цикл: OCR → Clean → RAG → LLM → JSON |
| `/recognize/upload` | POST | То же, но принимает файл (для Swagger) |
| `/recognize/{task_id}/status` | GET | Статус задачи распознавания |
| `/recognize/{task_id}/result` | GET | Результат (JSON или XML формат по выбору) |

#### `app/services/ocr_service.py`
Сервис для взаимодействия с внешним OCR API (`ocrbot.ru`):
- `create_task()` — Кодирует изображение в base64, отправляет POST на API, получает `task_id`
- `wait_for_task()` — Поллинг статуса задачи (до 60 секунд)
- `fetch_result()` — Получение результата в виде XML-байтов (base64-декодирование)
- `process_docs_directory()` — Пакетная обработка: сканирует `data/docs/*/image/`, создаёт XML в `data/docs/*/xml/`
- `process_directory()` — Альтернатива: обрабатывает файлы из `data/ocr/`

#### `app/services/cleaner_service.py`
Сервис для извлечения чистого текста из XML-результатов OCR:
- `_extract_raw_lines()` — Извлекает из XML-дерева строки типа `RIL_TEXTLINE`, считывает координаты (X, Y) и слова (`RIL_WORD`). Фильтрует по уровню уверенности (`cnf`)
- `_sort_and_group_lines()` — Сортирует строки по Y, группирует близкие по вертикали (threshold = 25px) в одну логическую строку, затем сортирует по X
- `parse_xml_bytes()` — Полный парсинг XML байтов в чистый текст
- `process_docs_directory()` — Пакетная обработка: `data/docs/*/xml/` → `data/docs/*/clean/`

#### `app/services/vector_service.py`
Сервис для работы с векторной базой данных Qdrant и Ollama Embeddings:
- **Подключения**: Qdrant (порт 6333), Ollama для генерации (порт 11434), Ollama для эмбеддингов (порт 11435)
- `vectorize_text()` — Преобразует текст в вектор через Ollama embedding-модель (обрезает до 3000 символов)
- `ensure_collection()` / `reset_collection()` — Управление коллекциями Qdrant (автоопределение размера вектора, автосброс при несовпадении размеров)
- `index_examples()` — Индексирует файлы из `data/examples/` (пары `*_input.txt` + `*_output.json`) в Qdrant с метаданными (тип документа, текст, JSON-выход)
- `search()` — Семантический поиск: векторизует запрос, ищет в Qdrant с опциональной фильтрацией по `doc_type`
- `reindex_all()` — Полная переиндексация: сбрасывает коллекцию и индексирует примеры заново
- `_detect_doc_type()` — Определение типа документа по имени файла (ключевые слова: passport, snils, driver, birth и т.д.)
- Сохраняет текущую embedding-модель в `data/rag/state.json`

#### `app/services/structuring_service.py`
Основной сервис RAG + LLM структурирования:
- `structure()` — Главный метод:
  1. **RAG**: Ищет ближайший пример в Qdrant по очищенному тексту → определяет тип документа
  2. **Шаблон**: Загружает JSON-шаблон для типа документа из `templates/`
  3. **Промпт**: Формирует сообщение для LLM:
     - Системная роль: «professional data extraction system»
     - JSON-схема для заполнения
     - Few-Shot пример (если найден через RAG)
     - Целевой OCR-текст
  4. **LLM**: Отправляет запрос в Ollama (стриминг), парсит JSON-ответ
  5. Поддерживает retry (до 3 попыток), настраиваемые temperature, num_ctx, timeout
  6. Подмена модели: если передан `llama3`, используется `qwen3:8b`

#### `app/services/recognition_service.py`
Оркестратор полного цикла распознавания (асинхронный):
- Хранилище задач в памяти (`_tasks: Dict`)
- `start_template_task()` — Создаёт фоновую задачу: base64 → сохранение → OCR → XML
- `start_recognition_task()` — Создаёт фоновую задачу: base64 → OCR → Clean → Structuring → JSON
- `get_task_status()` / `get_task_result()` — Опрос статуса и получение результата
- Использует `asyncio.create_task()` для fire-and-forget запуска, `asyncio.to_thread()` для синхронных OCR-вызовов
- Автоматически удаляет временные файлы после обработки

#### `app/services/benchmark_service.py`
Сервис для автоматического тестирования retrieval (Embedding-бенчмарк):
- `run()` — Запускает полный цикл:
  1. Создаёт отдельную коллекцию `benchmark_documents` в Qdrant
  2. Индексирует все примеры из `data/examples/`
  3. Проходит по всем файлам в `data/docs/*/clean/`
  4. Для каждого файла ищет ближайший пример → сравнивает предсказанный тип с ожидаемым
  5. Возвращает accuracy, количество правильных/ошибочных по каждому файлу, топ-3 альтернативы
- Поддерживает отмену (`stop()`)

#### `app/services/structuring_benchmark_service.py`
Сервис для тестирования качества LLM-структурирования (600 строк):
- `run()` — Прогон всех документов из `data/docs/*/clean/` через LLM:
  1. Переиндексирует примеры с выбранной embedding-моделью
  2. Для каждого файла: Clean → RAG → LLM → JSON
  3. Сравнивает с эталоном из `data/references/`
- **Метрики**: Accuracy, Precision, Recall, F1, CER (Levenshtein), Fuzzy Score (token_sort_ratio)
- `_flatten_dict()` — Рекурсивное разворачивание вложенных словарей для поэлементного сравнения
- `_calculate_field_metrics()` — Нормализация значений (uppercase, без пробелов и пунктуации) и поэлементное сравнение
- `run_multi()` — Последовательный запуск по нескольким моделям
- **Прогресс**: Live-отслеживание через `get_progress()` (текущий файл, стрим LLM-ответа)
- **Управление**: `stop()` для полной остановки, `skip_model()` для пропуска текущей модели, авто-пропуск при 3 ошибках подряд
- **Отчёты**: Сохранение в JSON (`data/benchmark/reports/`), CRUD операции (list, get, delete, clear)

---

### frontend/ — Веб-интерфейс

#### `frontend/index.html`
Одностраничное HTML-приложение с 5 панелями:
1. **Статус системы** — Количество OCR-файлов, очищенных файлов, векторов, текущая embedding-модель
2. **Пайплайн** — Кнопки для запуска OCR, очистки и индексации + конфигурация RAG (выбор модели, переиндексация)
3. **Поиск** — Текстовое поле для семантического поиска по документам
4. **Бенчмарк структурирования (LLM)** — Выбор моделей (чекбоксы), настройка параметров (temperature, context size, timeout), прогресс-бар, live-стрим ответа LLM, таблица результатов, история отчётов
5. **Бенчмарк Embedding** — Выбор embedding-модели, запуск, таблица результатов

#### `frontend/app.js`
JavaScript-логика UI (781 строка):
- Взаимодействие со всеми API-эндпоинтами через `fetch()`
- Рендеринг таблиц бенчмарков с цветовой индикацией (зелёный — OK, красный — ERR)
- Tooltips с подробным сравнением полей (expected vs actual)
- Polling прогресса LLM-бенчмарка (каждую секунду)
- Управление историей отчётов (загрузка, удаление, очистка, сортировка)
- Динамическая загрузка списков моделей при старте

#### `frontend/styles.css`
Стили UI (528 строк): сетка карточек, таблицы бенчмарков, прогресс-бар, tooltips, история отчётов, цветовое кодирование результатов.

---

### templates/ — JSON-шаблоны документов

JSON-схемы, определяющие структуру данных для каждого типа документа. Используются LLM как инструкция для заполнения.

| Файл | Тип документа | Ключевые поля |
|------|---------------|---------------|
| `passport_ru.json` | Паспорт РФ | серия, номер, ФИО, пол, дата/место рождения, кем выдан, дата выдачи, код подразделения, MRZ |
| `driver_license_ru.json` | Водительское удостоверение | номер, ФИО, дата/место рождения, дата выдачи/окончания, кем выдано, регион, категории |
| `snils.json` | СНИЛС | страховой номер, ФИО, дата/место рождения, пол, дата регистрации |
| `birth_certificate_ru.json` | Свидетельство о рождении | данные ребёнка (ФИО, дата/место рождения), данные отца и матери, запись акта, серия/номер |

---

### scripts/ — Утилиты

#### `scripts/ocr_fetcher.py`
Standalone-скрипт для пакетной отправки изображений на OCR. Содержит класс `ProtonOCR` — самодостаточный клиент для `ocrbot.ru`. Сканирует `data/ocr/`, находит изображения без corresponding XML, обрабатывает их и сохраняет результат.

#### `scripts/ocr_cleaner.py`
Standalone-скрипт для очистки XML-результатов OCR. Дублирует логику `CleanerService`, но работает напрямую с файлами в `data/ocr/`. Парсит XML, извлекает строки, сортирует по координатам, сохраняет чистый текст.

#### `scripts/generate_references.py`
Генерирует эталонные JSON-файлы для бенчмарков. Проходит по всем документам в `data/docs/*/clean/`, прогоняет через RAG + LLM и сохраняет результат в `data/references/` как `*-reference.json`. Используется для создания «золотого стандарта» при тестировании.

#### `scripts/check_vectors.py`
Диагностический скрипт для инспекции содержимого Qdrant: выводит количество точек в коллекции `documents`, фильтрует по `is_example`, показывает ID и filename каждой точки.

#### `scripts/debug_search.py`
Скрипт для отладки семантического поиска. Сравнивает результаты поиска по оригинальному тексту (с переносами строк) и по «схлопнутому» тексту (в одну строку). Помогает выявить проблемы с качеством embedding.

---

### data/ — Данные

| Директория | Содержание |
|------------|------------|
| `data/docs/` | Документы, организованные по типам: `passport/`, `snils/`, `driver_license/`, `birth_certificate/`. Каждый тип содержит поддиректории: `image/` (сканы), `xml/` (результаты OCR), `clean/` (очищенный текст) |
| `data/examples/` | Few-Shot примеры для RAG. Пары файлов: `*_input.txt` (очищенный OCR-текст) + `*_output.json` (эталонный JSON). По 2 примера для каждого типа документа |
| `data/references/` | Эталонные JSON для бенчмарков (генерируются через `scripts/generate_references.py`) |
| `data/ocr/` | Исторические OCR-файлы (альтернативное хранилище) |
| `data/rag/` | Хранилище Qdrant (том Docker) + `state.json` (текущая embedding-модель) |
| `data/benchmark/reports/` | Сохранённые JSON-отчёты бенчмарков |
| `data/temp/` | Временные файлы (base64 → изображения при обработке) |

---

### tests/ — Тесты

| Файл | Что тестирует |
|------|---------------|
| `test_api.py` | API-эндпоинты (OCR scan, clean, index, search, status) |
| `test_benchmark_service.py` | Embedding-бенчмарк (run, stop, evaluate) |
| `test_structuring_benchmark_service.py` | LLM-бенчмарк (метрики, field_metrics, flatten, CER, fuzzy, прогресс) |
| `test_structuring_service.py` | RAG + LLM структурирование |
| `test_vector_service.py` | Qdrant операции (index, search, reindex) |
| `test_cleaner.py` | Парсинг XML → чистый текст |
| `test_docs_pipeline.py` | Полный пайплайн (docs directory processing) |

---

## Установка и запуск

### Предварительные требования
- Python 3.10+
- Docker (для Qdrant)
- Ollama (установлен и запущен)
- API-ключ для [ocrbot.ru](https://ocrbot.ru)

### Шаги

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd diplom

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Настроить переменные окружения
cp .env.example .env
# Отредактировать .env: добавить OCR_API_KEY

# 5. Запустить Qdrant
docker-compose up -d

# 6. Запустить Ollama (два инстанса)
ollama serve                          # порт 11434 (генерация)
OLLAMA_HOST=0.0.0.0:11435 ollama serve  # порт 11435 (эмбеддинги)

# 7. Загрузить модели в Ollama
ollama pull nomic-embed-text:latest   # embedding-модель
ollama pull qwen3:8b                  # LLM-модель

# 8. Запустить приложение
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте http://localhost:8000 в браузере.

### CLI

```bash
python cli.py ocr                    # Запуск OCR для новых изображений
python cli.py clean                  # Очистка XML файлов
python cli.py index                  # Индексация в Qdrant
python cli.py search --query "текст" # Семантический поиск
```

---

## API Endpoints

Полная документация доступна по адресу: http://localhost:8000/api/v1/openapi.json

Swagger UI: http://localhost:8000/docs
