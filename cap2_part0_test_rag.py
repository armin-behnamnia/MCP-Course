import os
from fastapi import FastAPI, Query
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="LRAA Search Service")

# --- Configuration ---
PAPER_DIRECTORY = "./data/allowed"  # Put your PDFs here
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize Embeddings (runs locally)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def initialize_index():
    """Indexes all PDFs in the directory."""
    if not os.path.exists(PAPER_DIRECTORY):
        os.makedirs(PAPER_DIRECTORY)
        print(f"Created {PAPER_DIRECTORY}. Add your PDFs there.")
        return None

    # 1. Load all PDFs from the directory
    loader = PyPDFDirectoryLoader(PAPER_DIRECTORY)
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

# Global variable to hold our database
vector_db = initialize_index()

@app.get("/search")
async def search_papers(query: str, n: int = 3):
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

@app.post("/reindex")
async def reindex():
    """Manual trigger to refresh the index if you added new files."""
    global vector_db
    vector_db = initialize_index()
    return {"status": "Index updated successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8016)