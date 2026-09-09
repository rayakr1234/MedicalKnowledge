import os
from dotenv import load_dotenv
from src.vector_store import VectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-120b", relevance_threshold: float = 1.2):
        self.vectorstore = VectorStore(persist_dir)
        self.relevance_threshold = relevance_threshold
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.ingestion import load_all_documents
            docs = load_all_documents("data/raw")
            if docs:
                self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to the .env file or environment.")
        self.llm = ChatGroq(model=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        results: list | None = None,
    ) -> str:
        if results is None:
            results = self.vectorstore.query(query, top_k=top_k)

        relevant_results = [
            result
            for result in results
            if result["metadata"]
            and result["distance"] <= self.relevance_threshold
        ]
        context_parts = []
        for position, result in enumerate(relevant_results, start=1):
            metadata = result["metadata"]
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            context_parts.append(
                f"[Source {position}: {os.path.basename(source)}{page_label}]\n"
                f"{metadata.get('text', '').strip()}"
            )

        context = "\n\n".join(context_parts)
        if not context:
            return "No relevant documents found."
        prompt = f"""You are a medical information assistant.

                    Answer the user's question using ONLY the
                    provided context.

                    Rules:
                    1. Do not invent medical facts.
                    2. Do not provide diagnosis.
                    3. Do not prescribe treatment.
                    4. If the context does not contain enough information,
                       say that the information is not available.
                    5. Cite the source label, filename, or page used for the answer.
                    6. Clearly distinguish general information from
                        patient-specific medical advice.
                    Query: {query}

                    Context:
                   {context}

                   Answer:"""
        response = self.llm.invoke([prompt])
        return str(response.content)
