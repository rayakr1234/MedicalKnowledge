import os
import pickle
import faiss
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingModel
from src.chunk import ChunkingModel

class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: str="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.index = None
        self.metadata = []
        self.model= SentenceTransformer(self.embedding_model)
        print(f"[DEBUG] Initialized VectorStore with persist_dir={self.persist_dir} and embedding_model={self.embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        # Generate chunking for the documents
        chunking=ChunkingModel()
        chunks=chunking.chunk_documents(documents)
        embedding=EmbeddingModel()
        vectors=embedding.embed(documents)
        # Collecting metadata
        metadatas = [
            {**doc.metadata, "text": doc.page_content}
            for doc in chunks
        ]
        self.add_embeddings(np.array(vectors).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")
    
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if self.index is None:
            raise ValueError("Cannot save an empty vector store")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")
 
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("Cannot search an empty vector store")
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results
    
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

    