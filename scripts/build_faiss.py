import os, json, glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_DIR = os.environ.get("CORPUS_DIR", "./corpus")
INDEX_DIR = os.environ.get("INDEX_DIR", "./index")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

os.makedirs(INDEX_DIR, exist_ok=True)
texts, meta = [], []

for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt"))):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        continue
    texts.append(content)
    meta.append({"path": path, "chars": len(content)})

if not texts:
    raise SystemExit(f"No .txt files found in {CORPUS_DIR}. Add files and retry.")

print(f"Loaded {len(texts)} documents from {CORPUS_DIR}")

embedder = SentenceTransformer(MODEL_NAME)
emb = embedder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)  
index.add(emb)

faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Saved FAISS index to {INDEX_DIR}/faiss.index and metadata.json")

