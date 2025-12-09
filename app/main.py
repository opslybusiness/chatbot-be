"""
FastAPI main application for RAG Chatbot
"""
import logging
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Configure logging for Vercel (logs to stdout/stderr)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting RAG Chatbot API")

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

# Middleware to normalize double slashes and handle OPTIONS preflight early
class NormalizePathMiddleware(BaseHTTPMiddleware):
    """Normalize double slashes and ensure paths are correct, handle OPTIONS early"""
    async def dispatch(self, request: Request, call_next):
        # Fix double slashes in path before processing
        original_path = request.url.path
        if "//" in original_path:
            # Normalize the path by replacing all double slashes with single slash
            normalized_path = original_path.replace("//", "/")
            # Update the scope path
            request.scope["path"] = normalized_path
        
        # Handle OPTIONS requests early to prevent any redirects
        if request.method == "OPTIONS":
            # Create a response with CORS headers immediately
            response = Response()
            origin = request.headers.get("origin")
            if origin in origins:
                response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept, Origin, X-Requested-With"
            response.headers["Access-Control-Max-Age"] = "3600"
            return response
        
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


@app.get("/debug/embedding-config")
async def get_embedding_config():
    """Debug endpoint to check current embedding configuration"""
    from app.services.embedding_service import get_embedding_service
    import os
    
    service = get_embedding_service()
    config = {
        "provider": service.get_provider(),
        "dimension": service.get_embedding_dimension(),
        "env_vars": {
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "not set"),
            "JINA_EMBEDDING_MODEL": os.getenv("JINA_EMBEDDING_MODEL", "not set"),
            "JINA_EMBEDDING_TASK": os.getenv("JINA_EMBEDDING_TASK", "not set"),
            "JINA_EMBEDDING_DIMENSIONS": os.getenv("JINA_EMBEDDING_DIMENSIONS", "not set"),
            "JINA_API_KEY": "set" if os.getenv("JINA_API_KEY") else "not set",
        }
    }
    logger.info(f"[API] Embedding config: {config}")
    return config


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
    logger.info(f"[API] POST /chat/message - user_id={user_id}, message_length={len(request.message)}")
    try:
        response = await chat_service.process_message(
            message=request.message,
            user_id=user_id,
            session_id=request.session_id,
            use_rag=request.use_rag if hasattr(request, 'use_rag') else True
        )
        logger.info(f"[API] Successfully generated response, sources_count={len(response.get('sources', []))}")
        return ChatMessageResponse(
            session_id=request.session_id,
            message=response["message"],
            sources=response.get("sources", [])
        )
    except Exception as e:
        logger.error(f"[API] Error processing message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.options("/chat/session")
@app.post("/chat/session", response_model=ChatSessionResponse)
async def create_session(
    request: Request,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Get chat session info for the authenticated user.
    session_id is optional and only used for client-side reference (not stored in DB).
    All messages are matched by user_id only.
    Accepts optional JSON body with session_id or empty body.
    """
    try:
        session_id = None
        # Check if request has body content
        if request.method == "POST":
            try:
                body = await request.json()
                if body and "session_id" in body:
                    session_id = body.get("session_id")
            except:
                # No body or invalid JSON, continue with session_id = None
                pass
            
        session = await chat_service.create_session(user_id=user_id, session_id=session_id)
        return ChatSessionResponse(
            session_id=session["session_id"],
            created_at=session.get("created_at"),
            message_count=session.get("message_count", 0)
        )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@app.options("/chat/sessions")
@app.get("/chat/sessions")
async def list_sessions(
    user_id: str = Depends(get_user_id_from_token)
):
    """
    List all chat sessions for the authenticated user.
    Since messages are stored per user_id, this returns a single session representing the user's conversation.
    """
    try:
        sessions = await chat_service.list_sessions(user_id=user_id)
        # Return as list to match frontend expectations
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


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
        
        # Convert messages to ChatMessage format if they exist
        messages = session.get("messages", [])
        chat_messages = []
        if messages:
            from app.schemas import ChatMessage
            for msg in messages:
                # Support both 'sender' and 'role' fields
                sender = msg.get("role") or msg.get("sender") or msg.get("type", "user")
                chat_messages.append(ChatMessage(
                    sender=sender,
                    content=msg.get("content") or msg.get("message", ""),
                    timestamp=msg.get("timestamp") or session.get("created_at")
                ))
        
        return ChatSessionResponse(
            session_id=session["session_id"],
            created_at=session.get("created_at"),
            message_count=session.get("message_count", 0),
            messages=chat_messages if chat_messages else None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}", exc_info=True)
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
    metadata: Optional[str] = None,
    user_id: str = Depends(get_user_id_from_token)
):
    """
    Upload a document file for embedding creation and storage.
    Supports text files, PDFs, and other document formats.
    Documents are associated with the authenticated user.
    """
    try:
        result = await document_service.upload_and_process_file(file, user_id, metadata)
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
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_user_id_from_token)
):
    """Upload multiple documents at once. Documents are associated with the authenticated user."""
    results = []
    for file in files:
        try:
            result = await document_service.upload_and_process_file(file, user_id)
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
    top_k: int = 5,
    user_id: str = Depends(get_user_id_from_token)
):
    """Search for similar documents using vector similarity. Only returns documents for the authenticated user."""
    try:
        results = await document_service.search_documents(query, user_id, top_k)
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


@app.options("/documents/list")
@app.get("/documents/list")
async def list_documents(
    user_id: str = Depends(get_user_id_from_token)
):
    """List all uploaded documents for the authenticated user"""
    try:
        documents = await document_service.list_documents(user_id)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_user_id_from_token)
):
    """Delete a document and its embeddings. Users can only delete their own documents."""
    try:
        await document_service.delete_document(document_id, user_id)
        return JSONResponse(content={"status": "success", "message": f"Document {document_id} deleted"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

