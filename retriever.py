import json
import logging
from sqlalchemy import text
from sqlmodel import Session
from embeddings_util import get_engine, embed_texts

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def retrieve_similar_docs(query: str, top_k: int = 5):
    logger.info(f"[Retriever] Starting retrieval for query: '{query[:100]}...' (top_k={top_k})")
    
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
        
        # Step 3: Execute similarity search
        logger.info("[Retriever] Step 3: Executing similarity search...")
        sql = text("""
            SELECT id, content, metadata,
                   1 - (embedding <=> (:query_embedding)::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> (:query_embedding)::vector
            LIMIT :top_k;
        """)

        with Session(engine) as session:
            logger.debug(f"[Retriever] Executing SQL query with top_k={top_k}")
            results = session.execute(
                sql,
                {"query_embedding": query_vector, "top_k": top_k}
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
            logger.info(f"[Retriever] Best match similarity: {docs[0]['similarity']:.4f}")
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
