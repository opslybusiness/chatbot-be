"""
Chat service for handling chatbot interactions with RAG
"""
import os
import uuid
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

# Import existing modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from chat_memory import PostgresChatMessageHistory
from retriever import retrieve_similar_docs

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChatService:
    def __init__(self):
        """Initialize the chat service with LLM and RAG components"""
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            convert_system_message_to_human=True
        )
        
        # Create prompt template with RAG context
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful customer support bot. Answer all questions in conversation style and not too long. "
                "Use the provided context to answer questions accurately. If the context doesn't contain relevant "
                "information, say so politely."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create chain
        self.chain = self.prompt | self.model
        
        # Add message history support
        # We'll create the chain_with_memory dynamically per request with user_id
        self.chain = self.prompt | self.model
        
        # Message trimmer for long conversations
        self.trimmer = trim_messages(
            max_tokens=2000,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    def _get_session_history(self, user_id: str):
        """Get or create session history for user (user_id replaces session_id)"""
        return PostgresChatMessageHistory(user_id)
    
    def _get_session_history_factory(self, user_id: str):
        """Factory function to create a session history getter for a specific user"""
        def get_session_history(session_id: str):
            # session_id is ignored - we only use user_id for database operations
            return PostgresChatMessageHistory(user_id)
        return get_session_history

    async def create_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get or create chat session for user (session_id is optional client-side identifier, not stored in DB)"""
        # Generate session_id if not provided (only for client reference, not stored)
        if not session_id:
            session_id = f"user_{user_id}"
        
        # Initialize session by getting history (creates if doesn't exist)
        # Note: session_id is not used in database - only user_id is stored
        history = self._get_session_history(user_id)
        messages = history.messages
        
        # Calculate created_at from first message if exists
        created_at = datetime.now()
        if messages:
            # If we had timestamps in DB, use first message's timestamp
            created_at = datetime.now()
        
        return {
            "session_id": session_id,  # Returned for client reference only
            "created_at": created_at,
            "message_count": len(messages)
        }

    async def list_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """List all chat sessions for a user (since messages are stored per user_id, returns single session)"""
        try:
            # Get history by user_id
            history = self._get_session_history(user_id)
            messages = history.messages
            
            # Calculate created_at from first message (if exists)
            created_at = None
            if messages:
                # Try to get first message timestamp from database if available
                # For now, use current time as fallback
                created_at = datetime.now()
            
            # Generate a session_id based on user_id for consistency
            session_id = f"user_{user_id}"
            
            return [{
                "session_id": session_id,
                "id": session_id,  # Also include 'id' field for frontend compatibility
                "created_at": created_at,
                "date": created_at,  # Also include 'date' field for frontend compatibility
                "timestamp": created_at,  # Also include 'timestamp' field for frontend compatibility
                "message_count": len(messages)
            }]
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
            return []

    async def get_session(self, user_id: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get chat history for user (session_id is optional client-side identifier, not used in DB)"""
        try:
            # Generate session_id if not provided (only for client reference)
            if not session_id:
                session_id = f"user_{user_id}"
            
            # Get history by user_id only (session_id not stored in DB)
            history = self._get_session_history(user_id)
            messages = history.messages
            
            # Calculate created_at from first message
            created_at = datetime.now()
            if messages:
                # If we had timestamps in DB, use first message's timestamp
                created_at = datetime.now()
            
            # Convert to dict format - use both 'sender' and 'role' for compatibility
            message_list = []
            for msg in messages:
                sender_type = "user" if msg.type == "human" else "ai"
                message_list.append({
                    "sender": sender_type,  # For backward compatibility
                    "role": sender_type,  # Frontend expects 'role'
                    "type": sender_type,  # Also include 'type' for compatibility
                    "content": msg.content,
                    "message": msg.content,  # Also include 'message' field for frontend compatibility
                    "timestamp": created_at  # Use session created_at or calculate from DB if available
                })
            
            return {
                "session_id": session_id,  # Returned for client reference only
                "created_at": created_at,
                "date": created_at,  # Also include 'date' field
                "timestamp": created_at,  # Also include 'timestamp' field
                "message_count": len(messages),
                "messages": message_list,
                "history": message_list  # Also include 'history' field for frontend compatibility
            }
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}", exc_info=True)
            return None

    async def clear_session(self, user_id: str, session_id: Optional[str] = None) -> None:
        """Clear all chat history for a user (session_id is ignored - all messages cleared)"""
        # session_id is ignored - we clear all messages for the user
        history = self._get_session_history(user_id)
        history.clear()

    async def process_message(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response using RAG if enabled.
        
        Args:
            message: User message
            user_id: User ID from JWT token
            session_id: Optional client-side session identifier (ignored, only user_id is used)
            use_rag: Whether to use RAG for context retrieval
            
        Returns:
            Dictionary with 'message' (AI response) and 'sources' (retrieved documents)
        """
        import asyncio
        
        # Use user_id as session identifier for RunnableWithMessageHistory
        # session_id from client is ignored - we only use user_id for database operations
        config = {"configurable": {"session_id": user_id}}
        
        # Retrieve relevant documents if RAG is enabled
        logger.info(f"[ChatService] Processing message for user_id={user_id}, use_rag={use_rag}")
        logger.debug(f"[ChatService] Message: '{message[:100]}...'")
        
        sources = []
        context = ""
        
        if use_rag:
            logger.info("[ChatService] RAG enabled - starting document retrieval...")
            try:
                # Run synchronous retrieval in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                logger.info(f"[ChatService] Calling retrieve_similar_docs with query: '{message[:100]}...', user_id={user_id}")
                similar_docs = await loop.run_in_executor(
                    None, retrieve_similar_docs, message, 3, user_id
                )
                logger.info(f"[ChatService] Retrieved {len(similar_docs)} similar document(s)")
                
                if similar_docs:
                    sources = similar_docs
                    best_similarity = similar_docs[0].get('similarity', 0)
                    logger.info(f"[ChatService] Best match similarity: {best_similarity:.4f}")
                    
                    # Warn if similarity is too low
                    if best_similarity < 0.3:
                        logger.warning(f"[ChatService] ⚠️ LOW SIMILARITY: {best_similarity:.4f} - documents may not be relevant")
                        logger.warning("[ChatService] Consider checking if documents need to be re-embedded")
                    
                    # Format context from retrieved documents
                    context_parts = []
                    for i, doc in enumerate(similar_docs, 1):
                        similarity = doc.get('similarity', 0)
                        content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                        logger.info(f"[ChatService] Context doc {i}: similarity={similarity:.4f}, id={doc.get('id', 'N/A')}")
                        logger.info(f"[ChatService] Context doc {i} preview: {content_preview}")
                        context_parts.append(f"Context: {doc['content']}")
                    context = "\n\n".join(context_parts)
                    logger.info(f"[ChatService] Context length: {len(context)} characters")
                    
                    # Add context to the message
                    enhanced_message = f"{message}\n\nRelevant information:\n{context}"
                    logger.debug(f"[ChatService] Enhanced message length: {len(enhanced_message)} characters")
                else:
                    logger.warning("[ChatService] No similar documents found - proceeding without context")
                    enhanced_message = message
            except Exception as e:
                # If RAG fails, continue without context
                logger.error(f"[ChatService] RAG retrieval error: {str(e)}", exc_info=True)
                enhanced_message = message
        else:
            logger.info("[ChatService] RAG disabled - using original message")
            enhanced_message = message
        
        # Invoke the chain with message history
        try:
            logger.info("[ChatService] Creating chain with message history...")
            # Create chain with memory for this specific user
            # session_id is ignored - only user_id is used for database operations
            get_session_history = self._get_session_history_factory(user_id)
            chain_with_memory = RunnableWithMessageHistory(
                self.chain,
                get_session_history
            )
            
            logger.info("[ChatService] Invoking LLM chain...")
            # Run synchronous chain invocation in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: chain_with_memory.invoke(
                    [HumanMessage(content=enhanced_message)],
                    config=config
                )
            )
            
            ai_response = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"[ChatService] LLM response generated, length: {len(ai_response)} characters")
            logger.debug(f"[ChatService] Response preview: {ai_response[:200]}...")
            logger.info(f"[ChatService] Returning response with {len(sources)} source(s)")
            
            return {
                "message": ai_response,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"[ChatService] Error generating response: {str(e)}", exc_info=True)
            raise Exception(f"Error generating response: {str(e)}")

