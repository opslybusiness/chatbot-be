import json
import logging
from sqlalchemy import text
from sqlmodel import Session
from embeddings_util import get_engine, embed_texts

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def retrieve_similar_docs(query: str, top_k: int = 5, user_id: str = None):
    logger.info(f"[Retriever] Starting retrieval for query: '{query[:100]}...' (top_k={top_k}, user_id={user_id})")
    
    try:
        # Step 1: Generate embedding for query
        logger.info("[Retriever] Step 1: Generating query embedding...")
        query_vector = embed_texts([query])[0]  # single embedding
        logger.info(f"[Retriever] Query embedding generated, dimension: {len(query_vector)}")
        logger.debug(f"[Retriever] First 5 values of embedding: {query_vector[:5]}")
        
        # Step 2: Get database connection
        logger.info("[Retriever] Step 2: Getting database engine...")
        engine = get_engine()
        logger.info("[Retriever] Database engine obtained")
        
        # Step 3: Execute similarity search with user_id filtering
        logger.info("[Retriever] Step 3: Executing similarity search...")
        if user_id:
            # Filter documents by user_id stored in metadata
            sql = text("""
                SELECT id, content, metadata,
                       1 - (embedding <=> (:query_embedding)::vector) AS similarity
                FROM documents
                WHERE jsonb_extract_path_text(metadata::jsonb, 'user_id') = :user_id
                ORDER BY embedding <=> (:query_embedding)::vector
                LIMIT :top_k;
            """)
            params = {"query_embedding": query_vector, "top_k": top_k, "user_id": user_id}
            logger.info(f"[Retriever] Filtering documents by user_id: {user_id}")
        else:
            # For backward compatibility, allow queries without user_id (but log a warning)
            logger.warning("[Retriever] ⚠️ No user_id provided - retrieving documents from all users")
            sql = text("""
                SELECT id, content, metadata,
                       1 - (embedding <=> (:query_embedding)::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> (:query_embedding)::vector
                LIMIT :top_k;
            """)
            params = {"query_embedding": query_vector, "top_k": top_k}

        with Session(engine) as session:
            logger.debug(f"[Retriever] Executing SQL query with top_k={top_k}")
            results = session.execute(
                sql,
                params
            ).fetchall()
            logger.info(f"[Retriever] Database query returned {len(results)} result(s)")

        # Step 4: Process results
        logger.info("[Retriever] Step 4: Processing results...")
        docs = []
        for i, r in enumerate(results, 1):
            try:
                meta = r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata)
                doc = {
                    "id": r.id,
                    "content": r.content,
                    "metadata": meta,
                    "similarity": float(r.similarity)
                }
                docs.append(doc)
                logger.debug(f"[Retriever] Result {i}: id={r.id}, similarity={r.similarity:.4f}, content_preview={r.content[:50]}...")
            except Exception as e:
                logger.error(f"[Retriever] Error processing result {i}: {str(e)}")
                continue
        
        logger.info(f"[Retriever] Successfully retrieved {len(docs)} document(s)")
        if docs:
            best_similarity = docs[0]['similarity']
            logger.info(f"[Retriever] Best match similarity: {best_similarity:.4f}")
            
            # Warn if similarity is too low (likely embedding mismatch)
            if best_similarity < 0.3:
                logger.warning(f"[Retriever] ⚠️ LOW SIMILARITY WARNING: Best match is only {best_similarity:.4f} (4.7%)")
                logger.warning("[Retriever] This suggests possible issues:")
                logger.warning("[Retriever] 1. Documents may have been embedded with a different provider/model")
                logger.warning("[Retriever] 2. Embedding dimensions may not match")
                logger.warning("[Retriever] 3. Documents may not contain relevant information")
                logger.warning("[Retriever] 4. Consider re-embedding documents with current provider")
            
            # Log document previews for debugging
            for i, doc in enumerate(docs[:3], 1):
                content_preview = doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
                logger.info(f"[Retriever] Doc {i}: similarity={doc['similarity']:.4f}, id={doc['id']}")
                logger.info(f"[Retriever] Doc {i} content preview: {content_preview}")
        else:
            logger.warning("[Retriever] No documents retrieved - database may be empty or query doesn't match")
        
        return docs
    except Exception as e:
        logger.error(f"[Retriever] Error during retrieval: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    query = "are pets allowed?"
    results = retrieve_similar_docs(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['id']} (score={r['similarity']:.4f})")
        print(r['content'])
