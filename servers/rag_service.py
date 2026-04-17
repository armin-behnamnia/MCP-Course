import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI
from pathlib import Path

app = FastAPI(title="LRAA Automated Service")


# --- Configuration ---
DB_PATH = str(Path(os.path.dirname(__file__)).parent / "./chroma_db")
EMBEDDING_MODEL = str(Path(os.path.dirname(__file__)).parent / "./models/all-MiniLM-L6-v2")

print(f"DB path is {DB_PATH}")
print(f"Embedding model path is {EMBEDDING_MODEL}")

# Initialize Embeddings (runs locally)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def initialize_index(parent_directory: str):
    """Indexes all PDFs in the directory."""
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
        print(f"Created {parent_directory}. Add your PDFs there.")
        return None

    # 1. Load all PDFs from the directory
    loader = PyPDFDirectoryLoader(parent_directory)
    documents = loader.load()
    
    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create/Update Vector Store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    return vector_db

# --- Background Task Configuration ---


def search_papers(vector_db: Chroma, query: str, n: int = 3):
    """
    Returns the top n related chunks for a given query.
    """
    if not vector_db:
        return {"error": "No documents indexed. Add PDFs to the directory and restart."}
    
    # Perform similarity search
    results = vector_db.similarity_search_with_score(query, k=n)
    
    # Format response
    output = []
    for doc, score in results:
        output.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "score": score,
            "content": doc.page_content
        })
    
    return {"query": query, "results": output}