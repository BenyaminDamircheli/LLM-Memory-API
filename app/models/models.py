from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime

class Memory(BaseModel):
    """
    A memory is a piece of information that is stored in the memory engine.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the memory")
    content: str = Field(..., description="The content of the memory")
    embedding: Optional[List[float]] = Field(default=None, description="The embedding vector")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the memory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    namespace: Optional[str] = Field(default=None, description="The namespace/collection for the memory")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class CreateMemoryRequest(BaseModel):
    """
    Request model for creating a memory.
    """
    content: str = Field(..., description="The content of the memory")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the memory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    namespace: Optional[str] = Field(default=None, description="The namespace/collection for the memory")

class UpdateMemoryRequest(BaseModel):
    """
    Request model for updating a memory.
    """
    content: Optional[str] = Field(default=None, description="The content of the memory")
    tags: Optional[List[str]] = Field(default=None, description="Tags associated with the memory")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class SearchMemoryRequest(BaseModel):
    """
    Request model for searching memories.
    """
    query: str = Field(..., description="The query to search for")
    tags: Optional[List[str]] = Field(default=None, description="Tags to filter memories")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    limit: int = Field(default=10, description="The number of memories to return")
    sortBy: Optional[str] = Field(default=None, description="The field to sort by")

class SearchMemoryResponse(BaseModel):
    """
    Response model for searching memories.
    """
    memories: List[Memory] = Field(..., description="The memories found")
    similarity_scores: List[float] = Field(..., description="Similarity scores for each memory")



