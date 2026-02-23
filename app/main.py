import logging
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.endpoints import router as api_router
from app.core.config import settings

# Настройка логирования с метками времени
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("diplom")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

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
