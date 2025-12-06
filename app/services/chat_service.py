"""
Chat service for handling chatbot interactions with RAG
"""
import os
import uuid
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
            session_id = str(uuid.uuid4())
        
        # Initialize session by getting history (creates if doesn't exist)
        # Note: session_id is not used in database - only user_id is stored
        history = self._get_session_history(user_id)
        messages = history.messages
        
        return {
            "session_id": session_id,  # Returned for client reference only
            "created_at": datetime.now(),
            "message_count": len(messages)
        }

    async def get_session(self, user_id: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get chat history for user (session_id is optional client-side identifier, not used in DB)"""
        try:
            # Generate session_id if not provided (only for client reference)
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Get history by user_id only (session_id not stored in DB)
            history = self._get_session_history(user_id)
            messages = history.messages
            
            # Convert to dict format
            message_list = []
            for msg in messages:
                message_list.append({
                    "sender": "user" if msg.type == "human" else "ai",
                    "content": msg.content,
                    "timestamp": datetime.now()  # Note: You may want to store actual timestamps
                })
            
            return {
                "session_id": session_id,  # Returned for client reference only
                "created_at": datetime.now(),  # Note: Store actual creation time
                "message_count": len(messages),
                "messages": message_list
            }
        except Exception:
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
        sources = []
        context = ""
        
        if use_rag:
            try:
                # Run synchronous retrieval in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                similar_docs = await loop.run_in_executor(
                    None, retrieve_similar_docs, message, 3
                )
                if similar_docs:
                    sources = similar_docs
                    # Format context from retrieved documents
                    context_parts = []
                    for doc in similar_docs:
                        context_parts.append(f"Context: {doc['content']}")
                    context = "\n\n".join(context_parts)
                    
                    # Add context to the message
                    enhanced_message = f"{message}\n\nRelevant information:\n{context}"
                else:
                    enhanced_message = message
            except Exception as e:
                # If RAG fails, continue without context
                print(f"RAG retrieval error: {e}")
                enhanced_message = message
        else:
            enhanced_message = message
        
        # Invoke the chain with message history
        try:
            # Create chain with memory for this specific user
            # session_id is ignored - only user_id is used for database operations
            get_session_history = self._get_session_history_factory(user_id)
            chain_with_memory = RunnableWithMessageHistory(
                self.chain,
                get_session_history
            )
            
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
            
            return {
                "message": ai_response,
                "sources": sources
            }
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

