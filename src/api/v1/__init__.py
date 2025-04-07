from fastapi import APIRouter

# Create the main router
router = APIRouter(prefix="/v1")

# Import the route modules
from .chat import router as chat_router
from .conversations import router as conversations_router
from .tts import router as tts_router

# Include all routers
router.include_router(chat_router)
router.include_router(conversations_router)
router.include_router(tts_router)
