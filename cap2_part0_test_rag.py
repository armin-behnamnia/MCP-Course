import os
from fastapi import FastAPI, Query
from servers.rag_service import search_papers, initialize_index
from pathlib import Path

current_dir = Path(os.path.dirname(__file__))

app = FastAPI(title="LRAA Search Service")

vector_db = initialize_index(str(current_dir / "files" / "rag"))  # Index PDFs in the current directory

@app.get("/search")
async def search(query: str, n: int = 3):
    """
    Returns the top n related chunks for a given query.
    """
    
    return search_papers(vector_db, query, n)
    

@app.post("/reindex")
async def reindex():
    """Manual trigger to refresh the index if you added new files."""
    global vector_db
    vector_db = initialize_index()
    return {"status": "Index updated successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8016)