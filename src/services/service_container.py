import logging
from typing import Dict, Any, Optional, Type, Callable

logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Dependency injection container for managing service instances.
    Handles dependencies between services automatically, with improved
    circular dependency resolution.
    """

    def __init__(self):
        """Initialize the service container."""
        self._services = {}  # Type: Dict[str, Any]
        self._factories = {}  # Type: Dict[str, Callable]
        self._dependencies = {}  # Type: Dict[str, Dict[str, str]]
        self._initializing = set()  # Track services currently being initialized

    def register(self, name: str, instance: Any) -> None:
        """
        Register a service instance with the container.

        Args:
            name: Service name/key
            instance: Service instance
        """
        self._services[name] = instance
        logger.debug(f"Registered service: {name}")

    def register_factory(
            self,
            name: str,
            factory: Callable,
            dependencies: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a factory function to create a service on demand with dependencies.

        Args:
            name: Service name/key
            factory: Function that creates the service instance
            dependencies: Dict mapping parameter names to service names
        """
        self._factories[name] = factory
        if dependencies:
            self._dependencies[name] = dependencies
        logger.debug(f"Registered factory for service: {name}")

    def get(self, name: str) -> Any:
        """
        Get a service by name, initializing it if needed, with dependencies.
        Handles circular dependencies by partially initializing services.

        Args:
            name: Service name/key

        Returns:
            Service instance

        Raises:
            KeyError: If service is not registered
            ValueError: If there's a circular dependency that can't be resolved
        """
        # Return from services if already initialized
        if name in self._services:
            return self._services[name]

        # Detect circular dependencies
        if name in self._initializing:
            # For true circular dependencies, we'll return a partially initialized
            # service and fix the remaining dependencies later
            logger.warning(f"Circular dependency detected for {name}")
            return self._services.get(name)  # May be None or partially initialized

        # Try to initialize from factories
        if name in self._factories:
            # Mark this service as being initialized to detect circles
            self._initializing.add(name)

            try:
                # Get the factory
                factory = self._factories[name]

                # If there are dependencies, resolve them
                if name in self._dependencies:
                    kwargs = {}
                    for param_name, service_name in self._dependencies[name].items():
                        # Skip self-referencing dependencies - will be fixed later
                        if service_name == name:
                            logger.warning(f"Self-reference in {name}: param {param_name} -> {service_name}")
                            continue

                        # Recursively get dependent services
                        kwargs[param_name] = self.get(service_name)

                    # Create the service with dependencies
                    self._services[name] = factory(**kwargs)
                else:
                    # Create the service without dependencies
                    self._services[name] = factory()

                logger.debug(f"Initialized service: {name}")
                self._initializing.remove(name)
                return self._services[name]

            except Exception as e:
                self._initializing.remove(name)
                logger.error(f"Error initializing service {name}: {str(e)}")
                raise

        raise KeyError(f"Service not registered: {name}")

    def has(self, name: str) -> bool:
        """
        Check if a service is registered.

        Args:
            name: Service name/key

        Returns:
            True if service is registered, False otherwise
        """
        return name in self._services or name in self._factories

    def unregister(self, name: str) -> None:
        """
        Unregister a service.

        Args:
            name: Service name/key
        """
        if name in self._services:
            del self._services[name]
        if name in self._factories:
            del self._factories[name]
        if name in self._dependencies:
            del self._dependencies[name]
        if name in self._initializing:
            self._initializing.remove(name)

    def initialize(self, *names) -> None:
        """
        Initialize multiple services at once.

        Args:
            *names: Service names to initialize
        """
        for name in names:
            self.get(name)  # This will initialize if needed

    def connect_services(self, service_name: str, attribute_name: str, dependency_name: str) -> None:
        """
        Connect services after initialization to resolve circular dependencies.

        Args:
            service_name: The service that needs to be updated
            attribute_name: The attribute to set
            dependency_name: The dependency service to connect
        """
        if service_name in self._services and dependency_name in self._services:
            service = self._services[service_name]
            dependency = self._services[dependency_name]

            if hasattr(service, attribute_name):
                setattr(service, attribute_name, dependency)
                logger.info(f"Connected {dependency_name} to {service_name}.{attribute_name}")
            else:
                logger.warning(f"Service {service_name} has no attribute {attribute_name}")
        else:
            missing = []
            if service_name not in self._services:
                missing.append(service_name)
            if dependency_name not in self._services:
                missing.append(dependency_name)
            logger.warning(f"Cannot connect: services not initialized: {', '.join(missing)}")

    def reset(self) -> None:
        """Reset all services and factories."""
        self._services = {}
        self._factories = {}
        self._dependencies = {}
        self._initializing = set()


# Create global service container
services = ServiceContainer()


# Register all services
def initialize_services():
    """
    Initialize all services with proper dependency injection.
    Creates services and repositories with their dependencies properly wired.
    """
    # Import repository implementations
    from src.repositories.conversation import MongoConversationRepository
    from src.repositories.message import MongoMessageRepository
    from src.repositories.memory import MongoMemoryRepository
    from src.repositories.audio import MongoAudioRepository

    # Import service implementations
    from src.services.memory import MemoryService
    from src.services.chat import ChatService
    from src.services.conversation import ConversationService
    from src.services.recognition import SpeechRecognitionService
    from src.services.tts import TextToSpeechService
    from src.infrastracture.api_clients.external_api_client import ExternalAPIClient
    from src.core.utils.audio.audio_handling import AudioProcessor

    # Register AudioProcessor as a singleton first
    audio_processor = AudioProcessor()
    services.register("audio_processor", audio_processor)

    # Register API client
    services.register_factory("external_api_client", lambda: ExternalAPIClient())

    # Register repositories with audio processor
    services.register_factory(
        "audio_repository",
        lambda audio_processor: MongoAudioRepository(audio_processor=audio_processor),
        dependencies={"audio_processor": "audio_processor"}
    )
    services.register_factory("conversation_repository", lambda: MongoConversationRepository())
    services.register_factory("message_repository", lambda: MongoMessageRepository())
    services.register_factory("memory_repository", lambda: MongoMemoryRepository())

    # Register memory service first (without the summarizer yet)
    services.register_factory("memory_service", lambda: MemoryService())

    # Register other core services with dependencies
    # Register conversation service with dependencies
    services.register_factory(
        "conversation_service",
        lambda conversation_repo, message_repo, memory_repo, memory_service, external_api_client:
        ConversationService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            memory_repo=memory_repo,
            memory_service=memory_service,
            external_api_client=external_api_client
        ),
        dependencies={
            "conversation_repo": "conversation_repository",
            "message_repo": "message_repository",
            "memory_repo": "memory_repository",
            "memory_service": "memory_service",
            "external_api_client": "external_api_client"
        }
    )

    # Register chat service with dependencies
    services.register_factory(
        "chat_service",
        lambda external_api_client, memory_service, conversation_service:
        ChatService(
            external_api_client=external_api_client,
            memory_service=memory_service,
            conversation_service=conversation_service
        ),
        dependencies={
            "external_api_client": "external_api_client",
            "memory_service": "memory_service",
            "conversation_service": "conversation_service"
        }
    )

    # Register speech recognition service with dependencies and audio processor
    services.register_factory(
        "speech_recognition_service",
        lambda external_api_client, audio_repository, audio_processor:
        SpeechRecognitionService(
            external_api_client=external_api_client,
            audio_repository=audio_repository,
            audio_processor=audio_processor
        ),
        dependencies={
            "external_api_client": "external_api_client",
            "audio_repository": "audio_repository",
            "audio_processor": "audio_processor"
        }
    )

    # Register text-to-speech service with dependencies and audio processor
    services.register_factory(
        "tts_service",
        lambda external_api_client, audio_repository, audio_processor:
        TextToSpeechService(
            external_api_client=external_api_client,
            audio_repository=audio_repository,
            audio_processor=audio_processor
        ),
        dependencies={
            "external_api_client": "external_api_client",
            "audio_repository": "audio_repository",
            "audio_processor": "audio_processor"
        }
    )

    # Initialize all services
    services.initialize(
        "chat_service",
        "conversation_service",
        "speech_recognition_service",
        "tts_service"
    )

    # Resolve circular dependencies by explicitly connecting services
    # Connect memory service's summarizer to chat service
    memory_service = services.get("memory_service")
    chat_service = services.get("chat_service")
    memory_service.summarizer_service = chat_service

    logger.info("Service container initialized with all services and dependencies")


# Helper function to get a service with correct type
def get_service(name: str, service_type: Optional[Type] = None) -> Any:
    """
    Get a service with optional type checking.

    Args:
        name: Service name
        service_type: Optional type to check against

    Returns:
        Service instance

    Raises:
        TypeError: If service is not of the expected type
        KeyError: If service is not registered
    """
    service = services.get(name)
    if service_type and not isinstance(service, service_type):
        raise TypeError(f"Service {name} is not of type {service_type.__name__}")
    return service
