import hashlib
import tempfile
from pathlib import Path
import streamlit as st

from src.ingestion import load_all_documents
from src.search import RAGSearch

st.set_page_config(
    page_title="MedGuide RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MedGuide RAG")
st.caption("Evidence-grounded medical knowledge assistant")
st.warning(
    "For educational information only. This assistant does not diagnose conditions "
    "or replace a qualified healthcare professional."
)

@st.cache_resource(show_spinner="Loading the medical knowledge base...")
def load_assistant() -> RAGSearch:
    return RAGSearch()


def save_uploaded_files(uploaded_files: list) -> str:
    upload_dir = Path(tempfile.mkdtemp(prefix="medguide_uploads_"))
    for uploaded_file in uploaded_files:
        (upload_dir / uploaded_file.name).write_bytes(uploaded_file.getvalue())
    return str(upload_dir)


def get_uploaded_assistant(uploaded_files: list) -> RAGSearch:
    upload_key = hashlib.sha256(
        b"".join(file.getvalue() for file in uploaded_files)
    ).hexdigest()
    cached_key = st.session_state.get("upload_key")
    if cached_key != upload_key:
        upload_dir = save_uploaded_files(uploaded_files)
        uploaded_docs = load_all_documents(upload_dir)
        session_store = Path(tempfile.mkdtemp(prefix="medguide_store_"))
        assistant = RAGSearch(persist_dir=str(session_store))
        assistant.vectorstore.build_from_documents(uploaded_docs)
        st.session_state.upload_key = upload_key
        st.session_state.uploaded_assistant = assistant
    return st.session_state.uploaded_assistant


with st.sidebar:
    st.header("Knowledge sources")
    uploaded_files = st.file_uploader(
        "Add documents for this session",
        type=["pdf", "csv", "docx", "txt", "json", "xlsx"],
        accept_multiple_files=True,
        help="Uploaded files are indexed for this browser session only.",
    )
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)
    st.divider()
    st.caption("Built-in sources: data/raw")


def render_sources(results: list) -> None:
    relevant_results = [
        result
        for result in results
        if result.get("metadata") and result["distance"] <= 1.2
    ]
    if not relevant_results:
        st.info("No source passages met the relevance threshold.")
        return

    with st.expander(f"Sources ({len(relevant_results)})"):
        for position, result in enumerate(relevant_results, start=1):
            metadata = result["metadata"]
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            st.markdown(f"**{position}. {Path(source).name}{page_label}**")
            st.caption(metadata.get("text", "").strip())


question = st.chat_input("Ask a question about the medical documents...")

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            assistant = (
                get_uploaded_assistant(uploaded_files)
                if uploaded_files
                else load_assistant()
            )
            with st.spinner("Searching the knowledge base and preparing an answer..."):
                results = assistant.vectorstore.query(question, top_k=top_k)
                answer = assistant.search_and_summarize(
                    question,
                    top_k=top_k,
                    results=results,
                )
            st.markdown(answer)
            render_sources(results)
        except Exception as error:
            st.error("The assistant could not process that question.")
            st.exception(error)
else:
    st.info("Ask a question to search the built-in medical documents.")
