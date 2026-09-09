from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

class ChunkingModel:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        print(f"[DEBUG] Initialized ChunkingModel with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        chunks = self.text_splitter.split_documents(documents)
        print(f"[DEBUG] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
