from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

class BaseEmbeddingConfig(ABC):
    """
    Config for embeddings
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the embedding config

        Args:
            model (Optional[str], optional): The model to use for the embedding. Defaults to None.
            api_key (Optional[str], optional): The API key to use for the embedding. Defaults to None.
        """
        self.model = model
        self.api_key = api_key
    