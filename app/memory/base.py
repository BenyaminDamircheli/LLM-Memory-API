from abc import ABC, abstractmethod
from typing import List, Optional

from app.models.models import Memory

class BaseMemory(ABC):
    @abstractmethod
    def get(self, memory_id: str) -> Memory:
        """
        Get a memory by its ID.
        """
        pass

    @abstractmethod
    def get_all(self):
        """
        get all memories
        """
        pass
    