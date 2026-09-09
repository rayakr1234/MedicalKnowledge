# MedGuide RAG

MedGuide RAG is a Streamlit medical knowledge assistant built with Retrieval-Augmented Generation (RAG). It loads medical documents, splits them into chunks, creates semantic embeddings, searches those embeddings with FAISS, and sends relevant context to a Groq language model.

> **Medical disclaimer:** This project is for educational and research use only. It does not provide diagnosis, treatment, or patient-specific medical advice. Always consult a qualified healthcare professional.

## Features

- Ask questions through a Streamlit chat interface.
- Load PDF, CSV, DOCX, TXT, JSON, and XLSX documents.
- Upload additional documents for the current browser session.
- Use Sentence Transformers for local text embeddings.
- Use FAISS for similarity search.
- Display retrieved source passages and page numbers.
- Cache the loaded assistant to avoid reloading models on every interaction.
- Keep API keys, local documents, and generated vector indexes outside Git.

## Architecture

```text
Documents -> ingestion -> chunks -> embeddings -> FAISS index
                                                    ^
                                                    |
Question -> question embedding -> similarity search -+
                                                    |
                                                    v
                                  relevant context -> Groq LLM -> answer
```

## Project structure

```text
MedicalKnowledge/
├── app.py                  # Streamlit user interface
├── requirements.txt        # Python dependencies
├── .env.example            # Safe environment-variable template
├── src/
│   ├── ingestion.py        # Document loaders
│   ├── chunk.py            # Text splitting
│   ├── embedding.py        # Sentence Transformer embeddings
│   ├── vector_store.py     # FAISS index and metadata persistence
│   └── search.py            # Retrieval and Groq answer generation
├── data/raw/               # Local medical documents; ignored by Git
└── faiss_store/            # Generated FAISS files; ignored by Git
```

## Requirements

- Python 3.10 or newer
- A Groq API key
- Enough disk space for the embedding model and local documents

## Installation

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create your local environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your own Groq key:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or share the API key.

## Add knowledge documents

Place permitted local source documents inside:

```text
data/raw/
```

The application searches recursively for:

- `.pdf`
- `.csv`
- `.docx`
- `.txt`
- `.json`
- `.xlsx`

The repository intentionally ignores `data/` because source documents may be large, copyrighted, sensitive, or unsuitable for public redistribution. Only add documents to a public repository when you have permission to redistribute them.

You can also upload supported files from the Streamlit sidebar. Uploaded files are indexed in a temporary session store and are not added to the persistent project index.

## Run the application

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

If that port is busy, Streamlit chooses another available port or you can specify one:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## How the RAG pipeline works

1. `ingestion.py` loads supported files into LangChain documents.
2. `chunk.py` splits documents into approximately 1,000-character chunks with 200-character overlap.
3. `embedding.py` uses `all-MiniLM-L6-v2` to convert chunks into vectors.
4. `vector_store.py` stores vectors in a FAISS `IndexFlatL2` index and saves source metadata.
5. A user question is converted into a vector using the same embedding model.
6. FAISS returns the most similar chunks.
7. `search.py` filters results by relevance distance and places their text into a guarded prompt.
8. Groq generates an answer using the retrieved context.
9. Streamlit displays the answer and the retrieved source passages.

## Generated files

The first run may generate:

```text
faiss_store/faiss.index
faiss_store/metadata.pkl
```

These files are local build artifacts and are ignored by Git. Each developer or deployment environment can build its own index from its permitted documents.

## Security and privacy

- Rotate any API key that has ever been committed, pasted into a public issue, or exposed in logs.
- Keep `.env` local and use `.env.example` as the public template.
- Do not upload patient records, private documents, or sensitive data.
- Review document licenses before publishing medical PDFs.
- Do not treat generated answers as clinical advice.

## GitHub checklist

Before pushing:

```powershell
git status --short --ignored
git diff --check
git grep -n -I -E "gsk_[A-Za-z0-9]+|GROQ_API_KEY=" -- . ':!README.md' ':!.env.example'
```

The final command should return no live key. Confirm that `.env`, `.venv/`, `data/`, `faiss_store/`, and cache directories are ignored.

## License

Choose and add a license before publishing this project. If the included source documents are not yours to redistribute, do not publish them with the repository; document their sources and setup process instead.
