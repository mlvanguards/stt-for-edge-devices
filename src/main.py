import logging
import asyncio

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.db import manager as mongo_manager
from src.services.recognition import SpeechRecognitionService

from src.api.status import router as status_router
from src.api.v1 import router as router_v1


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_manager.init()

    # Start warm-up process in the background to not block startup
    asyncio.create_task(SpeechRecognitionService.warm_up_inference_api())
    logging.info("Hugging Face API warm-up initiated")

    try:
        yield
    finally:
        await mongo_manager.close()

# Create FastAPI application
fastapi_app = FastAPI(
    title="STT and Chat API with Conversation Memory, TTS, and MongoDB",
    version=settings.api.API_VERSION,
    description=
        "Speech-to-text transcription, conversation management with GPT models and short-term memory, "
        "and text-to-speech synthesis API optimized for edge devices.",
    root_path="/api",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app = CORSMiddleware(
    app=fastapi_app,
    allow_origins=settings.api.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.api.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.api.CORS_ALLOW_METHODS,
    allow_headers=settings.api.CORS_ALLOW_HEADERS,
)

# Include API routes
router = APIRouter()

router.include_router(status_router)
router.include_router(router_v1)

fastapi_app.include_router(router)