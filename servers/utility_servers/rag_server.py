from fastmcp import FastMCP
from dotenv import load_dotenv
from pathlib import Path
from .rag_service import search_papers, initialize_index
import os
import json
from apscheduler.schedulers.background import BackgroundScheduler
import re

PROXY = "http://192.168.10.2:3129" 

load_dotenv()

RAG_DIR: str = str(Path(os.path.dirname(__file__)).parent.parent / "./files/rag")

def scheduled_indexing(db_object: dict):
    """Wrapper to run the indexer in the background."""
    print("--- 🕒 Scheduled Task: Refreshing Vector Index ---")
    try:
        # if RAG_DB_DIR.exists() and RAG_DB_DIR.is_dir():
        #     shutil.rmtree(RAG_DB_DIR)
        db_object['ref'] = initialize_index(RAG_DIR)
        print("--- ✅ Indexing Complete ---")
    except Exception as e:
        print(f"--- ❌ Scheduled Indexing Failed: {e} ---")

# Initialize the scheduler
db_scheduler = BackgroundScheduler()

rag_db_object = {"ref": None}  # Mutable object to hold the vector_db reference for the scheduler
rag_db_object['ref'] = initialize_index(RAG_DIR)  # Initialize the RAG index at startup
# Add a job: runs every 24 hours
# You can change 'hours=24' to 'minutes=30' for more frequent updates during testing
db_scheduler.add_job(scheduled_indexing, kwargs={"db_object": rag_db_object}, trigger='interval', minutes=1, id='lraa_sync_job')
db_scheduler.start()

mcp = FastMCP(
    name="PDFFileServer",
    instructions=(
        "A read-only PDF file server. "
        "Use `list_pdf_files` to discover files by keyword. "
        "Use `read_pdf` to load a PDF and receive its content as Markdown. "
        "Restricted files require a valid access token."
    ),
)

def sanitize_input(text: str) -> str:
    return re.sub(r'[^\w\s\-\.,]', '', text).strip()


@mcp.tool(tags={'rag'}, name='search_research_papers')
def search_research_papers(query: str, n: int = 3) -> list[str]:
    """
    Performs a semantic similarity search across the local research repository (LRAA).
    
    Use this tool when the user asks questions about specific papers, technical 
    methodologies, or empirical results stored in the local PDF library. This tool 
    retrieves raw text chunks based on the conceptual meaning of the query, 
    not just keyword matching.

    Args:
        query (str): A detailed search string or specific question. For best results, 
                     use technical terms or full sentences (e.g., "latent space 
                     regularization in GANs").
        n (int): The number of top relevant text chunks to retrieve. Increase 'n' 
                 for complex topics requiring broader context. Default is 3.

    Returns:
        str: A concatenated string of text segments, each labeled with its 
             source filename and page number for auditing and citation purposes.
    """
    try:
        results = search_papers(rag_db_object['ref'], query, n)
        
        if not results or "results" not in results:
            return "Search completed: No relevant text chunks were found for this query."

        # # Building a structured response for the LLM's context window
        # formatted_output = [f"Found {len(results['results'])} relevant segments:\n"]
        formatted_output = []        
        for i, res in enumerate(results["results"]):
            content = res.get('content', '')
            content = sanitize_input(content)
            formatted_output.append(content)
            # formatted_output.append(
            #     {
            #         # "file": res.get('source', 'Unknown'),
            #         # "page": res.get('page', 'N/A'),
            #         "content": content,
            #         "score": res.get('score', 0)
            #     }
            # )
        return formatted_output
        
    except Exception as e:
        return f"Error accessing the research repository: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='http', port=8031)