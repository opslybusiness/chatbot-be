"""
FastAPI main application for RAG Chatbot
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
    DocumentSearchResponse,
    HealthResponse,
    FileUploadResponse
)
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.auth import get_user_id_from_token

app = FastAPI(
    title="RAG Chatbot API",
    description="FastAPI backend for RAG-based chatbot with document ingestion",
    version="1.0.0",
    redirect_slashes=False  # Disable automatic slash redirects to prevent CORS preflight issues
)


origins = [
    "https://marketing-minds-three.vercel.app",
    "https://www.opslybusiness.me",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]

# Middleware to normalize double slashes in paths
class NormalizePathMiddleware(BaseHTTPMiddleware):
    """Normalize double slashes and ensure paths are correct"""
    async def dispatch(self, request: Request, call_next):
        # Fix double slashes in path before processing
        original_path = request.url.path
        if "//" in original_path:
            # Create a new scope with normalized path
            normalized_path = original_path.replace("//", "/")
            scope = dict(request.scope)
            scope["path"] = normalized_path
            # Reconstruct the request URL with normalized path
            if normalized_path != original_path:
                # Create new request with corrected path
                # Note: We modify the scope which will affect downstream handlers
                request.scope["path"] = normalized_path
        response = await call_next(request)
        return response

# Add path normalization middleware first (before CORS)
app.add_middleware(NormalizePathMiddleware)

# CORS middleware - must be added after path normalization but before other middleware
# This handles preflight OPTIONS requests properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*", "Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)


# Initialize services
chat_service = ChatService()
document_service = DocumentService()


@app.get("/", response_model=HealthResponse)
@app.get("", response_model=HealthResponse)  # Handle both with and without trailing slash
async def root():
    """Root endpoint - health check"""
    return HealthResponse(status="ok", message="RAG Chatbot API is running")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="ok", message="Service is healthy")


@app.post("/chat/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Send a message to the chatbot and receive a response.
    The chatbot uses RAG to retrieve relevant context from documents.
    All messages are matched by user_id only (session_id is optional and ignored).
    """
    try:
        response = await chat_service.process_message(
            message=request.message,
            user_id=user_id,
            session_id=request.session_id,
            use_rag=request.use_rag if hasattr(request, 'use_rag') else True
        )
        return ChatMessageResponse(
            session_id=request.session_id,
            message=response["message"],
            sources=response.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/chat/session", response_model=ChatSessionResponse)
async def create_session(
    session_id: Optional[str] = None,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Get chat session info for the authenticated user.
    session_id is optional and only used for client-side reference (not stored in DB).
    All messages are matched by user_id only.
    """
    try:
        session = await chat_service.create_session(user_id=user_id, session_id=session_id)
        return ChatSessionResponse(
            session_id=session["session_id"],
            created_at=session.get("created_at"),
            message_count=session.get("message_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@app.get("/chat/session/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Get chat session history for the authenticated user.
    session_id is optional client-side identifier - all messages are matched by user_id only.
    """
    try:
        session = await chat_service.get_session(user_id=user_id, session_id=session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return ChatSessionResponse(
            session_id=session["session_id"],
            created_at=session.get("created_at"),
            message_count=session.get("message_count", 0),
            messages=session.get("messages", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")


@app.delete("/chat/session/{session_id}")
async def clear_session(
    session_id: str,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Clear all chat history for the authenticated user.
    session_id is optional client-side identifier - all messages are cleared for the user.
    """
    try:
        await chat_service.clear_session(user_id=user_id, session_id=session_id)
        return JSONResponse(content={"status": "success", "message": f"Session {session_id} cleared"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.post("/documents/upload", response_model=FileUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """
    Upload a document file for embedding creation and storage.
    Supports text files, PDFs, and other document formats.
    """
    try:
        result = await document_service.upload_and_process_file(file, metadata)
        return FileUploadResponse(
            file_id=result["file_id"],
            filename=result["filename"],
            chunks_created=result["chunks_created"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")


@app.post("/documents/upload-multiple", response_model=List[FileUploadResponse])
async def upload_multiple_documents(
    files: List[UploadFile] = File(...)
):
    """Upload multiple documents at once"""
    results = []
    for file in files:
        try:
            result = await document_service.upload_and_process_file(file)
            results.append(FileUploadResponse(
                file_id=result["file_id"],
                filename=result["filename"],
                chunks_created=result["chunks_created"],
                status="success"
            ))
        except Exception as e:
            results.append(FileUploadResponse(
                file_id=None,
                filename=file.filename,
                chunks_created=0,
                status="error",
                error=str(e)
            ))
    return results


@app.get("/documents/search", response_model=List[DocumentSearchResponse])
async def search_documents(
    query: str,
    top_k: int = 5
):
    """Search for similar documents using vector similarity"""
    try:
        results = await document_service.search_documents(query, top_k)
        return [
            DocumentSearchResponse(
                id=doc["id"],
                content=doc["content"],
                similarity=doc["similarity"],
                metadata=doc.get("metadata", {})
            )
            for doc in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


@app.get("/documents/list")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = await document_service.list_documents()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    try:
        await document_service.delete_document(document_id)
        return JSONResponse(content={"status": "success", "message": f"Document {document_id} deleted"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

