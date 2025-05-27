from abc import ABC, abstractmethod
from typing import List, Optional

class BaseEmbeddingModel(ABC):
    def __init__(self, api_key: Optional[str], model: str):
        """
        Abstract base class for embedding models.

        Args:
            api_key: The API key for the embedding model.
            model: The model to use for the embedding.
        """
        self.api_key = api_key
        self.model = model
        self._validate_model()
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Embed a text string

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding of the text.
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings

        Args:
            texts: The list of text to embed.

        Returns:
            A list of lists of floats representing the embeddings of the text.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The name of the embedding model.
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        The dimension of the embedding.
        """
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported model names."""
        pass
    
    def _validate_model(self) -> None:
        """Validate that the model is supported."""
        if self.model not in self.supported_models:
            raise ValueError(f"Model {self.model} not supported. Supported models include: {self.supported_models}")

    
    