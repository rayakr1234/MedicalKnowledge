from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, TextLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# Loading all type of documents
def load_all_documents(data_dir: str) -> List[Any]:
    # Project root path folder
    data_path=Path(data_dir).resolve()
    print(f"[Debug] Data path: {data_path}")
    documents=[]


     # Loading PDF Files
    pdf_files=list(data_path.glob('**/*.pdf'))
    print(f"Found {len(pdf_files)} PDF Files")
    for pdf_file in pdf_files:
        print(f"Loading PDF File : {pdf_file}")
        try :
           loader=PyMuPDFLoader(str(pdf_file))
           loaded= loader.load()
           print(f"Loaded {pdf_file}")
           documents.extend(loaded)
        except Exception as e:
            print(f"Failed to load the pdf file {pdf_file}")
    
    # Loading Text Files
    text_files=list(data_path.glob("**/*.txt"))
    print(f"Found {len(text_files)} text files")
    for text_file in text_files :
        try : 
            loader=TextLoader(str(text_file))
            loaded=loader.load()
            print(f"Loaded {text_file}")
            documents.extend(loaded)
        except Exception as e :
            print(f"Error occured during loading text file")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file), jq_schema='.[]')
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

docs=load_all_documents("data/raw")
print(docs)