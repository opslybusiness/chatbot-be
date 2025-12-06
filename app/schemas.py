"""
Pydantic schemas for request/response models
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    message: str


class ChatMessageRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Optional client-side session ID (ignored, only used for response)")
    use_rag: bool = Field(True, description="Whether to use RAG for context retrieval")


class ChatMessageResponse(BaseModel):
    message: str = Field(..., description="AI response message")
    session_id: Optional[str] = Field(None, description="Client-side session identifier")
    sources: Optional[List[Dict[str, Any]]] = Field(default=[], description="Source documents used for RAG")


class ChatMessage(BaseModel):
    sender: str = Field(..., description="'user' or 'ai'")
    content: str
    timestamp: Optional[datetime] = None


class ChatSessionResponse(BaseModel):
    session_id: Optional[str] = Field(None, description="Client-side session identifier")
    created_at: Optional[datetime] = None
    message_count: int = 0
    messages: Optional[List[ChatMessage]] = None


class DocumentSearchResponse(BaseModel):
    id: str
    content: str
    similarity: float = Field(..., ge=0, le=1, description="Similarity score between 0 and 1")
    metadata: Optional[Dict[str, Any]] = None


class FileUploadResponse(BaseModel):
    file_id: Optional[str] = None
    filename: str
    chunks_created: int = 0
    status: str = Field(..., description="'success' or 'error'")
    error: Optional[str] = None

