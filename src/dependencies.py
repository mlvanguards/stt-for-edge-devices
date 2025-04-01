from fastapi import Depends
from src.db import manager as mongo_manager
from src.repositories.conversation import ConversationRepository
from src.repositories.message import MessageRepository
from src.repositories.audio import AudioRepository
from src.repositories.memory import MemoryRepository
from src.gateways.huggingface import HuggingFaceGatewayClient
from src.gateways.openai import OpenAIGatewayClient
from src.gateways.elevenlabs import ElevenLabsGatewayClient
from src.services.chat import ChatService
from src.services.conversation import ConversationService
from src.services.memory import MemoryService
from src.services.recognition import SpeechRecognitionService
from src.services.tts import TextToSpeechService
from src.utils.audio.audio_handling import AudioProcessorMainApp

async def get_db():
    db = await mongo_manager.get_db()
    return db

async def get_audio_processor():
    return AudioProcessorMainApp()

def get_huggingface_client():
    return HuggingFaceGatewayClient()

def get_openai_client():
    return OpenAIGatewayClient()

def get_elevenlabs_client():
    return ElevenLabsGatewayClient()

async def get_conversation_repository(db=Depends(get_db)):
    return ConversationRepository(db)

async def get_message_repository(db=Depends(get_db)):
    return MessageRepository(db)

async def get_audio_repository(db=Depends(get_db), audio_processor=Depends(get_audio_processor)):
    return AudioRepository(db, audio_processor=audio_processor)

async def get_memory_repository(db=Depends(get_db)):
    return MemoryRepository(db)

async def get_memory_service():
    return MemoryService(summarizer_service=None)

async def get_conversation_service(
    conversation_repo=Depends(get_conversation_repository),
    message_repo=Depends(get_message_repository),
    memory_repo=Depends(get_memory_repository),
    memory_service=Depends(get_memory_service),
    openai_client=Depends(get_openai_client)
):
    return ConversationService(conversation_repo, message_repo, memory_repo, memory_service, openai_client)

async def get_chat_service(
    memory_service=Depends(get_memory_service),
    conversation_service=Depends(get_conversation_service),
    openai_client=Depends(get_openai_client)
):
    chat_service = ChatService(memory_service, conversation_service)
    chat_service.external_api_client = openai_client
    return chat_service

async def get_speech_recognition_service(
    huggingface_client=Depends(get_huggingface_client),
    audio_repository=Depends(get_audio_repository),
    audio_processor=Depends(get_audio_processor)
):
    return SpeechRecognitionService(huggingface_client, audio_repository, audio_processor)

async def get_tts_service(
    elevenlabs_client=Depends(get_elevenlabs_client),
    audio_repository=Depends(get_audio_repository),
    audio_processor=Depends(get_audio_processor)
):
    return TextToSpeechService(elevenlabs_client, audio_repository, audio_processor)
