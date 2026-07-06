import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer, util



df = pd.read_csv("./data/pool_a_kb.csv", index_col=0)


ids = [f"kb_{i}" for i in df.index]
texts_to_embed = df["body"].tolist()
documents = df["answer"].tolist()
metadatas = [{"queue": str(row.queue), "original_index": int(row.Index), "priority": str(row.priority), "body_preview": str(row.body)[:100]} for row in df.itertuples()]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=64)


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="kb_tickets")


batch_size = 1000

for i in range(0, len(ids), batch_size):
    collection.add(
        ids=ids[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size].tolist(),
        documents=documents[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )
    print(f"Added batch {i} to {i+batch_size}")
print(collection.count())
