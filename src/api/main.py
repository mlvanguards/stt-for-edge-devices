import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router as api_router
from src.config.settings import settings
from src.services.service_container import initialize_services, services
from src.database.mongo import mongodb_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    Separates app creation from startup event handling.

    Returns:
        Configured FastAPI application
    """
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

    # Include all API routes at once from registry
    app.include_router(api_router)

    # Add middleware for request time measurement
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """
        Middleware to measure and log request processing time.
        Also adds the processing time as a response header.
        """
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Log request info for non-health-check endpoints
        if not request.url.path == "/":
            logger.info(f"Request {request.method} {request.url.path} processed in {process_time:.4f}s")

        return response

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
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # Define startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Initialize the application on startup"""
        logger.info("Starting application initialization...")

        try:
            # Log database configuration
            mongodb_db = settings.db.MONGODB_DB
            logger.info(f"Using database: {mongodb_db}")

            # Mask sensitive parts of MongoDB URI for logging
            uri = settings.db.MONGODB_URI
            if uri:
                masked_uri = re.sub(
                    r"mongodb(\+srv)?://[^:]+:[^@]+@", "mongodb\\1://***:***@", uri
                )
                logger.info(f"Found MONGODB_URI: {masked_uri}")

            # Initialize MongoDB connection using the connection manager
            connected = await mongodb_connection.connect()

            if connected:
                logger.info(f"Successfully connected to MongoDB database: {mongodb_db}")
            else:
                logger.error("Failed to connect to MongoDB database")

            # Initialize all services
            initialize_services()
            logger.info("Services initialized")

            # Start warm-up process in the background to not block startup
            speech_recognition_service = services.get("speech_recognition_service")
            asyncio.create_task(speech_recognition_service.warm_up_inference_api())
            logger.info("Hugging Face API warm-up initiated")

        except Exception as e:
            logger.error(f"Error during startup: {str(e)}")

        logger.info(f"{settings.api.API_TITLE} v{settings.api.API_VERSION} started successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup resources on shutdown"""
        logger.info("Application shutdown initiated...")

        # Close MongoDB connection using the connection manager
        await mongodb_connection.close()
        logger.info("MongoDB connection closed")

        logger.info("Application shutdown complete")

    return app


# Create the FastAPI application instance
app = create_application()
