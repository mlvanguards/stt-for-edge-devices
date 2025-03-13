import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
from src.config.settings import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ALLOW_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS
)
from src.core.database import MongoDB
from src.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Include API routes
app.include_router(api_router)


# Health check endpoint
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Speech-to-text, Chat API, and Text-to-speech with MongoDB is running",
        "version": API_VERSION
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logging.info("Starting application initialization...")

    # Start MongoDB connection in background to not block startup
    import asyncio
    asyncio.create_task(MongoDB.connect())

    logging.info(f"{API_TITLE} v{API_VERSION} started successfully")

@app.get("/create-test-conversation")
async def create_test_conversation():
    """Create a test conversation to initialize the database"""
    try:
        conversation_id = str(uuid.uuid4())
        await MongoDB.create_conversation(
            conversation_id,
            "This is a test conversation",
            "default-voice-id"
        )
        await MongoDB.add_message(
            conversation_id,
            "user",
            "This is a test message"
        )
        return {"status": "success", "conversation_id": conversation_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    # Close MongoDB connection
    await MongoDB.close()
    logging.info("Application shutdown complete")

