from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Any
from src.chunk import ChunkingModel

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"[DEBUG] Loaded embedding model: {model_name}")
    def embed(self, documents: List[Any]) -> np.ndarray:
        chunking_model = ChunkingModel()
        chunks = chunking_model.chunk_documents(documents)
        text=[chunk.page_content for chunk in chunks]
        embeddings=self.model.encode(text, show_progress_bar=True)
        print(f"[DEBUG] Generated embeddings for {len(chunks)} chunks, Embedding shape: {embeddings.shape}")
        return embeddings