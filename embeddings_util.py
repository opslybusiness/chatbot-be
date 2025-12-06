# embedding_utils.py
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from sqlmodel import create_engine, Session
from sqlalchemy import text

# Import the new embedding service
# Add parent directory to path to import from app/services
import sys
from pathlib import Path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app.services.embedding_service import get_embedding_service, embed_texts

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# --- EMBEDDING SERVICE ---
# Use the new embedding service which supports both Jina and Gemini
# The provider is determined by EMBEDDING_PROVIDER env var ("jina" or "gemini")
_embedding_service = None

def get_embedding_service_instance():
    """Get the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = get_embedding_service()
    return _embedding_service

# Keep embed_texts function for backward compatibility
# It now uses the embedding service which can switch between providers

# --- ENGINE ---
_engine = None
def get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set in env")
        
        # Ensure we use psycopg (sync driver) instead of asyncpg
        # SQLAlchemy may auto-detect async drivers, so we explicitly use psycopg
        db_url = DATABASE_URL
        
        # Handle different URL formats
        if db_url.startswith("postgresql+asyncpg://"):
            # Replace asyncpg with psycopg
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
        elif db_url.startswith("postgresql://") and "+" not in db_url:
            # Replace postgresql:// with postgresql+psycopg:// for explicit sync driver
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif db_url.startswith("postgres://"):
            # Handle postgres:// alias
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
        
        # Create engine with connection pooling
        # Use psycopg (psycopg3) or psycopg2
        # Enable connection pooling with proper settings to prevent connection drops
        pool_settings = {
            "pool_size": 5,  # Number of connections to maintain
            "max_overflow": 10,  # Additional connections allowed beyond pool_size
            "pool_recycle": 3600,  # Recycle connections after 1 hour
            "pool_pre_ping": True,  # Test connections before using them (handles dropped connections)
            "pool_reset_on_return": "commit",  # Reset connections when returned to pool
        }
        try:
            _engine = create_engine(db_url, **pool_settings)
        except Exception as e:
            # If psycopg fails, try psycopg2
            if "psycopg" in db_url:
                db_url = db_url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
                _engine = create_engine(db_url, **pool_settings)
            else:
                raise e
    return _engine

# --- DB OPERATIONS ---
def upsert_documents(docs: List[Dict], batch_size: int = 100):
    engine = get_engine()  # reuse same engine every time
    with Session(engine) as session:
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d["content"] for d in batch]
            vectors = embed_texts(texts)
            
            for doc, vec in zip(batch, vectors):
                stmt = text(
                    """
                    INSERT INTO documents (id, content, metadata, embedding, created_at)
                    VALUES (:id, :content, :metadata, :embedding, now())
                    ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        created_at = now();
                    """
                )
                session.execute(
                    stmt,
                    {
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": json.dumps(doc.get("metadata") or {}),
                        "embedding": vec,
                    },
                )
        session.commit()

# --- DEMO / SANITY TEST ---
if __name__ == "__main__":
    sample = [
        {"id": "doc1", "content": "How to change my flight date?", "metadata": {"source": "faq"}},
        {"id": "doc2", "content": "Baggage allowance rules for economy tickets.", "metadata": {"source": "faq"}},
    ]
    upsert_documents(sample)
    print("Upserted sample docs")
