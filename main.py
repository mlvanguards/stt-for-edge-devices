import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.config.settings import settings
from src.services.recognition import SpeechRecognitionService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI application
app = FastAPI(
    title=settings.api.API_TITLE,
    version=settings.api.API_VERSION,
    description=settings.api.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.api.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.api.CORS_ALLOW_METHODS,
    allow_headers=settings.api.CORS_ALLOW_HEADERS,
)

# Include API routes
app.include_router(api_router)


# Health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Speech-to-text, Chat API, and Text-to-speech with MongoDB is running",
        "version": settings.api.API_VERSION,
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logging.info("Starting application initialization...")

    try:
        # Load MongoDB connection settings
        logging.info(f"Using database: {settings.db.MONGODB_DB}")

        # Print environment variables (masked) for debugging
        import re

        uri = settings.db.MONGODB_URI
        if uri:
            masked_uri = re.sub(
                r"mongodb(\+srv)?://[^:]+:[^@]+@", "mongodb\\1://***:***@", uri
            )
            logging.info(f"Found MONGODB_URI: {masked_uri}")

        # Initialize MongoDB connection
        from src.database.mongo import MongoDB

        connected = await MongoDB.connect()

        if connected:
            logging.info(
                f"Successfully connected to MongoDB database: {settings.db.MONGODB_DB}"
            )
        else:
            logging.error("Failed to connect to MongoDB database")

        # Start warm-up process in the background to not block startup
        asyncio.create_task(SpeechRecognitionService.warm_up_inference_api())
        logging.info("Hugging Face API warm-up initiated")

    except Exception as e:
        logging.error(f"Error during startup: {str(e)}")

    logging.info(
        f"{settings.api.API_TITLE} v{settings.api.API_VERSION} started successfully"
    )
