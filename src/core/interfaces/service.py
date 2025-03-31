from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class IMemoryService(ABC):
    """Interface for conversation memory service"""

    @abstractmethod
    def optimize_conversation_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize the conversation history for the LLM.

        Args:
            messages: The full conversation history

        Returns:
            An optimized list of messages
        """
        pass


class ISummarizerService(ABC):
    """Interface for conversation summarization"""

    @abstractmethod
    async def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of conversation messages.

        Args:
            messages: List of conversation messages to summarize

        Returns:
            Summary text or empty string if failed
        """
        pass


class IChatService(ABC):
    """Interface for chat completion services"""

    @abstractmethod
    async def get_chat_completion(
            self,
            prompt: str,
            conversation_id: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a response from a chat completion model.

        Args:
            prompt: User's input text
            conversation_id: Optional conversation ID to fetch history
            conversation_history: Optional explicit conversation history
            model: Optional model override
            temperature: Optional temperature setting
            max_tokens: Optional max tokens setting

        Returns:
            Dict with response and metrics
        """
        pass

    @abstractmethod
    async def process_chat_with_conversation(
            self,
            conversation_id: str,
            user_message: str,
            model: Optional[str] = None,
            temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message within a conversation context.

        Args:
            conversation_id: The conversation ID
            user_message: The user's message
            model: Optional model override
            temperature: Optional temperature setting

        Returns:
            Dict with response and conversation data
        """
        pass


class IConversationService(ABC):
    """Interface for conversation management"""

    @abstractmethod
    async def create_conversation(
            self,
            system_prompt: Optional[str] = None,
            voice_id: Optional[str] = None,
            stt_model_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new conversation.

        Args:
            system_prompt: Optional system prompt
            voice_id: Optional voice ID for TTS
            stt_model_id: Optional STT model ID

        Returns:
            The created conversation or None if failed
        """
        pass

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation ID

        Returns:
            The conversation if found, None otherwise
        """
        pass

    @abstractmethod
    async def add_message(
            self,
            conversation_id: str,
            role: str,
            content: str,
            timestamp: Optional[str] = None,
            importance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add a message to a conversation.

        Args:
            conversation_id: The conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            timestamp: Optional timestamp
            importance: Optional importance score

        Returns:
            The created message if successful, None otherwise
        """
        pass

    @abstractmethod
    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            List of messages
        """
        pass

    @abstractmethod
    async def extract_conversation_context(self, conversation_id: str) -> List[Dict]:
        """
        Extract formatted conversation context for LLM.

        Args:
            conversation_id: The conversation ID

        Returns:
            List of formatted messages
        """
        pass

    @abstractmethod
    async def summarize_memory(self, conversation_id: str) -> bool:
        """
        Create or update memory summary for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            True if summary created/updated, False otherwise
        """
        pass

class ISpeechRecognitionService(ABC):
    """Interface for speech recognition service"""

    @abstractmethod
    async def process_audio_file(
            self,
            audio_content: bytes,
            content_type: str,
            model_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            store_audio: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process an audio file for speech recognition.

        Args:
            audio_content: Raw audio bytes
            content_type: MIME type
            model_id: Optional model ID
            conversation_id: Optional conversation ID
            store_audio: Whether to store audio

        Returns:
            List of transcription segments
        """
        pass

    @abstractmethod
    def clean_transcription(self, transcriptions: List[Dict[str, Any]]) -> str:
        """
        Clean and format transcription results.

        Args:
            transcriptions: List of transcription segments

        Returns:
            Cleaned transcription text
        """
        pass


class ITextToSpeechService(ABC):
    """Interface for text-to-speech service"""

    @abstractmethod
    async def synthesize_speech(
            self,
            text: str,
            voice_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            return_base64: bool = True
    ) -> Dict[str, Any]:
        """
        Convert text to speech.

        Args:
            text: Text to convert
            voice_id: Optional voice ID
            conversation_id: Optional conversation ID
            return_base64: Whether to return base64 audio

        Returns:
            Dict with audio data and metadata
        """
        pass

    @abstractmethod
    async def get_available_voices(self) -> Dict[str, Any]:
        """
        Get available voices.

        Returns:
            Dict with voices information
        """
        pass


class IExternalAPIClient(ABC):
    """Interface for external API interactions."""

    @abstractmethod
    async def call_openai_api(
            self,
            messages: list,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call OpenAI API for chat completion.

        Args:
            messages: List of chat messages
            model: Optional model name
            temperature: Optional temperature value
            max_tokens: Optional token limit

        Returns:
            Response with success status and data
        """
        pass

    @abstractmethod
    async def call_huggingface_api(
            self,
            model_id: str,
            audio_content: bytes,
            content_type: str,
            max_retries: Optional[int] = None,
            backoff_factor: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call Hugging Face API for speech recognition.

        Args:
            model_id: HF model identifier
            audio_content: Audio bytes
            content_type: MIME type
            max_retries: Optional retry count
            backoff_factor: Optional backoff multiplier

        Returns:
            Response with transcription or error
        """
        pass

    @abstractmethod
    async def call_elevenlabs_api(
            self,
            text: str,
            voice_id: str,
            model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call ElevenLabs API for text-to-speech.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            model_id: Optional model identifier

        Returns:
            Response with audio content or error
        """
        pass

    @abstractmethod
    async def get_elevenlabs_voices(self) -> Dict[str, Any]:
        """
        Get available voices from ElevenLabs API.

        Returns:
            Response with voices data or error
        """
        pass
