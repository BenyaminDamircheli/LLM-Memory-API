import os
from typing import List, Optional
from openai import AsyncOpenAI
import asyncio
from .base import BaseEmbeddingModel

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        super().__init__(api_key, model)
    
    async def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed text: {str(e)}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        if not texts:
            return []
            
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to embed batch: {str(e)}")
    
    @property
    def model_name(self) -> str:
        return self.model
    
    @property
    def embedding_dimension(self) -> int:
        return self._model_dimensions[self.model]
    
    @property
    def supported_models(self) -> List[str]:
        return list(self._model_dimensions.keys())