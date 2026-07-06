import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("./data/pool_b_eval.csv", index_col=0)


ids = [f"kb_{i}" for i in df.index]
texts_to_embed = df["body"].tolist()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texßts_to_embed, show_progress_bar=True, batch_size=64)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="kb_tickets")
print(texts_to_embed[0])
print(collection.query(query_embeddings=embeddings[0], n_results=3))

