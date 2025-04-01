from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.db import manager as mongo_manager
from src.dependencies import (
    get_huggingface_client,
    get_audio_repository,
    get_audio_processor,
    get_db
)
from src.services.recognition import SpeechRecognitionService
from src.api.v1 import router as api_router

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context to initialize and clean up resources.
    This function manually resolves dependencies required for the startup tasks.
    """
    # Initialize MongoDB connection
    mongo_manager.init()
    try:
        hf_client = get_huggingface_client()
        db = await get_db()
        audio_processor = await get_audio_processor()
        audio_repo = await get_audio_repository(db=db, audio_processor=audio_processor)
        sr_service = SpeechRecognitionService(hf_client, audio_repo, audio_processor)

        # Schedule the warm-up coroutine in the background
        asyncio.create_task(sr_service.warm_up_inference_api())
        logger.info("Hugging Face API warm-up initiated")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}")
    try:
        yield
    finally:
        pass


def create_app() -> FastAPI:
    """
    Factory function for creating the FastAPI application.
    This function configures middleware, exception handlers, routers,
    and adds a health-check endpoint.
    """
    app = FastAPI(
        title=settings.api.API_TITLE,
        version=settings.api.API_VERSION,
        description=settings.api.API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Set up CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.CORS_ALLOW_ORIGINS,
        allow_credentials=settings.api.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.api.CORS_ALLOW_METHODS,
        allow_headers=settings.api.CORS_ALLOW_HEADERS,
    )

    # Include API routers
    app.include_router(api_router)

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # Middleware to add process time header and log request processing time
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"{request.method} {request.url.path} processed in {process_time:.4f}s")
        return response

    # Health-check endpoint
    @app.get("/")
    async def root():
        return {
            "status": "healthy",
            "message": "Speech-to-text, Chat API, and Text-to-speech with MongoDB is running",
            "version": settings.api.API_VERSION,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
