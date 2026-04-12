import logging
import sys
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.endpoints import router as api_router
from app.api.external import router as external_router
from app.core.config import settings

# Настройка логирования с метками времени
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level, logging.DEBUG))

# Создаем обработчик для файла
file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

# Создаем обработчик для консоли
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

logger = logging.getLogger("diplom")
logger.setLevel(getattr(logging, log_level, logging.DEBUG))

from contextlib import asynccontextmanager
from app.db.database import engine, Base
from app.services.vector_service import VectorService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Auto-index examples on startup
    try:
        logger.info("Running automatic indexation of examples...")
        vs = VectorService()
        indexed = await vs.index_examples()
        logger.info(f"Successfully checked and indexed examples: {indexed}")
    except Exception as e:
        logger.error(f"Error during initial indexing: {e}")

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["Internal API"])
app.include_router(external_router, prefix=settings.API_V1_STR, tags=["External API"])

# Mount static files (Frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    logger.info("Accessing root index.html")
    return FileResponse("frontend/index.html")

@app.get("/app.js")
async def read_js():
    return FileResponse("frontend/app.js")

@app.get("/styles.css")
async def read_css():
    return FileResponse("frontend/styles.css")
