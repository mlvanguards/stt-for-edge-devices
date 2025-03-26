from fastapi import APIRouter

# Create a router to include all other routers
router = APIRouter()

# Import routers
from .conversations import router as conversation_router
from .chat import router as chat_router
from .tts import router as tts_router
from .api_keys import router as api_keys_router

# Include all routers with their prefixes
router.include_router(api_keys_router)  # Include the API keys route
router.include_router(conversation_router, prefix="")
router.include_router(chat_router, prefix="")
router.include_router(tts_router, prefix="")