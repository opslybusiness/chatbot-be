# ingest_docs.py
# Example script to ingest local docs into pgvector store

import os
from embeddings_util import embed_texts, get_engine,upsert_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter



docs = []
chunks=[]

policy_path = "Rules.txt"
if os.path.exists(policy_path):
    with open("rules.txt", "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # max number of words/characters per chunk
    chunk_overlap=50,    # overlap between chunks
    separators=["\n\n", "\n", ".", "!", "?"]  # split hierarchy
    )

    chunks = splitter.split_text(text)
    
else:
    print("policy.txt not found in current directory.")



# prepare docs for ingestion
docs = [{"id": f"policy_{i}", "content": c, "metadata": {"source": "policy.txt"}} for i, c in enumerate(chunks)]
upsert_documents(docs)
print("Ingest complete. Chunks:", len(chunks))

