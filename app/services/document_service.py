"""
Document service for handling file uploads and embeddings
"""
import os
import uuid
from typing import Dict, List, Optional, Any
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import existing modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from embeddings_util import upsert_documents
from retriever import retrieve_similar_docs


class DocumentService:
    def __init__(self):
        """Initialize document service"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?"]
        )

    async def upload_and_process_file(
        self,
        file: UploadFile,
        metadata: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload and process a file, creating embeddings and storing in database.
        
        Args:
            file: Uploaded file
            metadata: Optional metadata string (JSON format)
            
        Returns:
            Dictionary with file processing results
        """
        try:
            # Read file content
            content = await file.read()
            
            # Decode text content
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings or handle binary files
                raise ValueError(f"File {file.filename} is not a valid text file")
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            # Parse metadata if provided
            file_metadata = {}
            if metadata:
                import json
                try:
                    file_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    pass
            
            # Prepare documents for ingestion
            file_id = str(uuid.uuid4())
            docs = [
                {
                    "id": f"{file_id}_chunk_{i}",
                    "content": chunk,
                    "metadata": {
                        "source": file.filename,
                        "file_id": file_id,
                        "chunk_index": i,
                        **file_metadata
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Upsert documents (creates embeddings and stores in DB)
            # Run in thread pool to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, upsert_documents, docs)
            
            return {
                "file_id": file_id,
                "filename": file.filename,
                "chunks_created": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")

    async def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, retrieve_similar_docs, query, top_k
            )
            return results
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")

    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents (grouped by file_id).
        
        Returns:
            List of document metadata
        """
        try:
            import asyncio
            from sqlmodel import Session, select, text
            from embeddings_util import get_engine
            
            def _list_docs():
                engine = get_engine()
                with Session(engine) as session:
                    # Get unique file_ids from metadata
                    result = session.execute(
                        text("""
                            SELECT DISTINCT 
                                jsonb_extract_path_text(metadata::jsonb, 'file_id') as file_id,
                                jsonb_extract_path_text(metadata::jsonb, 'source') as filename,
                                COUNT(*) as chunk_count
                            FROM documents
                            WHERE metadata::jsonb ? 'file_id'
                            GROUP BY file_id, filename
                            ORDER BY filename
                        """)
                    ).fetchall()
                    
                    documents = []
                    for row in result:
                        documents.append({
                            "file_id": row.file_id,
                            "filename": row.filename,
                            "chunk_count": row.chunk_count
                        })
                    
                    return documents
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _list_docs)
        except Exception as e:
            raise Exception(f"Error listing documents: {str(e)}")

    async def delete_document(self, document_id: str) -> None:
        """
        Delete a document and all its chunks from the database.
        
        Args:
            document_id: File ID or document ID to delete
        """
        try:
            import asyncio
            from sqlmodel import Session, text
            from embeddings_util import get_engine
            
            def _delete_doc():
                engine = get_engine()
                with Session(engine) as session:
                    # Delete by file_id (all chunks) or by specific document id
                    if document_id.startswith("_chunk_"):
                        # Delete specific chunk
                        session.execute(
                            text("DELETE FROM documents WHERE id = :doc_id"),
                            {"doc_id": document_id}
                        )
                    else:
                        # Delete all chunks for a file_id
                        session.execute(
                            text("""
                                DELETE FROM documents 
                                WHERE jsonb_extract_path_text(metadata::jsonb, 'file_id') = :file_id
                            """),
                            {"file_id": document_id}
                        )
                    session.commit()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _delete_doc)
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")

