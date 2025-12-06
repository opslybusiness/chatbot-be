import json
from sqlalchemy import text
from sqlmodel import Session
from embeddings_util import get_engine, embed_texts

def retrieve_similar_docs(query: str, top_k: int = 5):
    engine = get_engine()
    query_vector = embed_texts([query])[0]  # single embedding

    sql = text("""
        SELECT id, content, metadata,
               1 - (embedding <=> (:query_embedding)::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> (:query_embedding)::vector
        LIMIT :top_k;
    """)

    with Session(engine) as session:
        results = session.execute(
            sql,
            {"query_embedding": query_vector, "top_k": top_k}
        ).fetchall()

    docs = []
    for r in results:
        meta = r.metadata if isinstance(r.metadata, dict) else json.loads(r.metadata)
        docs.append({
            "id": r.id,
            "content": r.content,
            "metadata": meta,
            "similarity": float(r.similarity)
        })
    return docs


if __name__ == "__main__":
    query = "are pets allowed?"
    results = retrieve_similar_docs(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['id']} (score={r['similarity']:.4f})")
        print(r['content'])
