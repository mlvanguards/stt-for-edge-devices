from fastapi import APIRouter
from .conversations import router as conversation_router
from .chat import router as chat_router
from .tts import router as tts_router

# Create a router to include all other routers
router = APIRouter()

# Include all routers with their prefixes
router.include_router(conversation_router, prefix="")
router.include_router(chat_router, prefix="")
router.include_router(tts_router, prefix="")