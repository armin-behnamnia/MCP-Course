from fastmcp import FastMCP
from utils import _validate_token, _find_pdfs, _read_pdf_core, _parse_bold_headers, _extract_section_content, _generate_mini_summary
from dotenv import load_dotenv
from pathlib import Path
from rag_service import search_papers, initialize_index
import os
from typing import Optional
import json
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import re
from pydantic import BaseModel, Field

PROXY = "http://192.168.10.2:3129" 




load_dotenv()

ALLOWED_DIR: Path = Path(os.environ.get("ALLOWED_DIR", "")).resolve()
RESTRICTED_DIR: Path = Path(os.environ.get("RESTRICTED_DIR", "")).resolve()
RESTRICTED_TOKEN: str = os.environ.get("RESTRICTED_TOKEN", "")
RAG_DIR: str = str(Path(os.path.dirname(__file__)).parent / "./files/rag")


                # 'publisher': result['publisher'],
                # 'doi': result['DOI'],
                # 'source': result['source'],
                # 'title': result['title'][0],
                # 'author': result['author'],
                # 'year': result['created']['date-parts'][0][0]

class ArticleInfoContent(BaseModel):
    publisher: str = Field("", description="The publisher of the article")
    doi: str = Field("", description="The DOI of the article")
    source: str = Field("", description="The source of the article metadata")
    title: str = Field("", description="The official title of the article")
    author: list[dict] = Field([], description="The authors of the article")
    year: int = Field(0, description="The year of publication")
    citations: int = Field(0, description="The number of citations")

class ArticleInfo(BaseModel):
    status: str = Field("ok", description="API response status")
    content: ArticleInfoContent = Field(..., description="The metadata content of the article")
    error: str = Field("", description="Error message if the article info is invalid")


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

# # Initialize the scheduler
# db_scheduler = BackgroundScheduler()

# rag_db_object = {"ref": None}  # Mutable object to hold the vector_db reference for the scheduler
# rag_db_object['ref'] = initialize_index(RAG_DIR)  # Initialize the RAG index at startup
# # Add a job: runs every 24 hours
# # You can change 'hours=24' to 'minutes=30' for more frequent updates during testing
# db_scheduler.add_job(scheduled_indexing, kwargs={"db_object": rag_db_object}, trigger='interval', minutes=1, id='lraa_sync_job')
# db_scheduler.start()

mcp = FastMCP(
    name="PDFFileServer",
    instructions=(
        "A read-only PDF file server. "
        "Use `list_pdf_files` to discover files by keyword. "
        "Use `read_pdf` to load a PDF and receive its content as Markdown. "
        "Restricted files require a valid access token."
    ),
)

@mcp.resource(
    "config://server",
    name="ServerConfig",
    mime_type="application/json",
)
def resource_server_config() -> str:
    return '{"name": "pdfserver"}'

@mcp.tool(
    name="list_pdf_files",
    description=(
        "Search for PDF files by keyword in their filename (case-insensitive). "
        "Searches both the allowed and restricted folders. "
        "Returns a list of objects each with 'id', 'folder', and 'filename'. "
        "Pass an empty string to list all PDF files across both folders. "
        "Use the returned 'id' and 'folder' values with read_pdf."
    ),
    tags={"pdf", "search"},
)
def list_pdf_files(keyword: str = "", token: str = None) -> list[dict]:
    """
    Parameters
    ----------
    keyword : str
        Case-insensitive substring to match against filenames.
        Empty string returns every PDF in both folders.

    Returns
    -------
    list[dict]
        Each dict: {'id': str, 'folder': 'allowed'|'restricted', 'filename': str}
    """
    results: list[dict] = []

    for rel in _find_pdfs(ALLOWED_DIR, keyword):
        results.append({
            "id":       rel,
            "folder":   "allowed",
            "filename": Path(rel).name,
        })
    if token is not None:
        _validate_token(token)
        for rel in _find_pdfs(RESTRICTED_DIR, keyword):
            results.append({
                "id":       rel,
                "folder":   "restricted",
                "filename": Path(rel).name,
            })

    return results

@mcp.tool(
    name="read_pdf",
    description=(
        "Read a PDF file and return its full content as Markdown text. "
        "For 'allowed' files, no token is needed. "
        "For 'restricted' files, supply the correct token. "
        "Obtain valid file_id and folder values from list_pdf_files or "
        "from the catalog://allowed resource."
    ),
    tags={"pdf", "read"},
)
def read_pdf(
    file_id: str,
    folder: str,
    token: Optional[str] = None,
) -> str:
    """
    Parameters
    ----------
    file_id : str
        Relative path of the PDF (as returned by list_pdf_files).
    folder : str
        'allowed' or 'restricted'.
    token : str, optional
        Required when folder is 'restricted'.

    Returns
    -------
    str
        Full PDF content converted to GitHub-flavoured Markdown.
    """
    return _read_pdf_core(file_id=file_id, folder=folder, token=token)

@mcp.tool(
    name="extract_headers",
    description=(
        "Extract all bold headers from a PDF document. "
        "Headers are lines whose entire text is wrapped in double stars "
        "(**like this**). Returns an ordered list of header strings, "
        "stripped of the surrounding stars. "
        "Use the returned header strings with extract_section to retrieve "
        "the content beneath a specific header. "
        "Internally calls read_pdf, so the same file_id/folder/token rules apply."
    ),
    tags={"pdf", "headers"},
)
def extract_headers(
    file_id: str,
    folder: str,
    token: Optional[str] = None,
) -> list[str]:
    """
    Parameters
    ----------
    file_id : str
        Relative path of the PDF (as returned by list_pdf_files).
    folder : str
        'allowed' or 'restricted'.
    token : str, optional
        Required when folder is 'restricted'.

    Returns
    -------
    list[str]
        Ordered list of header texts found in the document, with the
        surrounding double stars removed.
        Example: ['Introduction', 'Methods', 'Results']
    """
    markdown = _read_pdf_core(file_id=file_id, folder=folder, token=token)
    return _parse_bold_headers(markdown)


@mcp.tool(
    name="extract_section",
    description=(
        "Extract the content that follows a specific bold header in a PDF. "
        "The header must match exactly one of the headers returned by "
        "extract_headers (comparison is case-insensitive and whitespace-tolerant). "
        "Content runs from the line after the matched header up to (but not "
        "including) the next bold header, or end-of-document. "
        "Internally calls read_pdf, so the same file_id/folder/token rules apply."
    ),
    tags={"pdf", "section"},
)
def extract_section(
    file_id: str,
    header: str,
    folder: str,
    token: Optional[str] = None,
) -> str:
    """
    Parameters
    ----------
    file_id : str
        Relative path of the PDF (as returned by list_pdf_files).
    header : str
        The header whose content you want, exactly as returned by
        extract_headers (without the surrounding double stars).
    folder : str
        'allowed' or 'restricted'.
    token : str, optional
        Required when folder is 'restricted'.

    Returns
    -------
    str
        The Markdown content between the matched header and the next bold
        header (or end of document), with leading/trailing blank lines stripped.

    Raises
    ------
    ValueError
        If no header matching *header* is found in the document.
    """
    markdown = _read_pdf_core(file_id=file_id, folder=folder, token=token)
    return _extract_section_content(markdown, header)


@mcp.tool(tags={"context_reshaping"})
def summarize_filtered_sections(keyword: str, section_target: str, token: Optional[str] = None, max_summaries: int = 2) -> list[dict]:
    """
        Performs a targeted cross-document search and generates a concise synthesis.
        
        Use this tool when you need to compare how a specific topic (keyword) is 
        addressed across multiple documents within a specific structural context 
        (e.g., comparing 'Methodology' or 'Future Work' across several papers).

        Args:
            keyword: The specific term, technology, or concept to search for within 
                    the section text (case-insensitive).
            section_target: The exact name of the section to target (e.g., 'Introduction', 
                            'Abstract', 'Conclusion', 'Results').
            token: Optional access token for restricted documents.  

        Returns:
            A formatted string containing the source filename and a 1-2 sentence 
            summary of the relevant section for every document where the keyword 
            was found. Returns a 'not found' message if no matches occur.

        Example:
            If you want to know how different papers introduce 'Reinforcement Learning', 
            call: search_and_summarize_sections(keyword="RL", section_target="Introduction")
    """
    pdf_files = list_pdf_files(keyword="", token=token)
    results = []
    for pdf in pdf_files:
        markdown = _read_pdf_core(file_id=pdf['id'], folder=pdf['folder'], token=token)
        
        try:
            section_text = _extract_section_content(markdown, section_target)
        except ValueError:
            continue  # Section not found, skip to next document    
        if section_text and keyword.lower() in section_text.lower():
            # 3. Generate a concise summary
            # In a real RAG setup, you'd call a small LLM completion here.
            # For this MCP tool, we'll simulate the logic or use a helper.
            summary = _generate_mini_summary(section_text) 
            
            results.append({
                "source": pdf['id'],
                "summary": summary
            })
        if len(results) >= max_summaries:
            break
            
    return results

# @mcp.tool(tags={'rag'}, name='search_research_papers')
# def search_research_papers(query: str, n: int = 3) -> str:
#     """
#     Performs a semantic similarity search across the local research repository (LRAA).
    
#     Use this tool when the user asks questions about specific papers, technical 
#     methodologies, or empirical results stored in the local PDF library. This tool 
#     retrieves raw text chunks based on the conceptual meaning of the query, 
#     not just keyword matching.

#     Args:
#         query (str): A detailed search string or specific question. For best results, 
#                      use technical terms or full sentences (e.g., "latent space 
#                      regularization in GANs").
#         n (int): The number of top relevant text chunks to retrieve. Increase 'n' 
#                  for complex topics requiring broader context. Default is 3.

#     Returns:
#         str: A concatenated string of text segments, each labeled with its 
#              source filename and page number for auditing and citation purposes.
#     """
#     try:
#         results = search_papers(rag_db_object['ref'], query, n)
        
#         if not results or "results" not in results:
#             return "Search completed: No relevant text chunks were found for this query."

#         # Building a structured response for the LLM's context window
#         formatted_output = [f"Found {len(results['results'])} relevant segments:\n"]
        
#         for i, res in enumerate(results["results"], 1):
#             formatted_output.append(
#                 {
#                     "file": res.get('source', 'Unknown'),
#                     "page": res.get('page', 'N/A'),
#                     "content": res.get('content', ''),
#                     "score": res.get('score', 0)
#                 }
#             )
#         return json.dumps(formatted_output)
        
#     except Exception as e:
#         return f"Error accessing the research repository: {str(e)}"


# # Security: Strict Regex for DOI validation (Prevents Injection)
# DOI_PATTERN = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', re.I)

def sanitize_input(text: str) -> str:
    return re.sub(r'[^\w\s\-\.,]', '', text).strip()

def sanitize_data(data: str) -> str:
    """Removes potential script tags or malicious characters from API responses."""
    if not data: 
        return ""
    return re.sub(r'<[^>]*?>', '', data).strip()


@mcp.tool()
def validate_and_fetch_metadata(title: str) -> ArticleInfo:
    """
    Searches a paper title and fetches official metadata from Crossref.
    Handles malicious input and sanitizes external responses.
    
    Args:
        title (str): the title string of the paper.
    Returns:
        str: the serialized dictionary of official_title, publisher, year of the paper
    """
    title = sanitize_input(title)
    print(f"Searching for title: {title}")
    params = {"query.title": title, "rows": 1}
    try:
        response = requests.get("https://api.crossref.org/works", timeout=5, params=params, proxies={"http": PROXY, "https": PROXY})
    except requests.RequestException as e:
        return ArticleInfo(
            status="error",
            content=ArticleInfoContent(),
            error=f"Failed to fetch metadata: {str(e)}"
        )
        #{"error": f"Failed to fetch metadata: {str(e)}"}
    data = sanitize_data(response.text)
    try:
        data = json.loads(data)
    except json.JSONDecodeError as e:
        return ArticleInfo(
            status="error",
            content=ArticleInfoContent(),
            error=f"Failed to parse metadata: {str(e)}"
        )
    status = data.get('status', 'error')
    message = data.get('message', dict())
    results = message.get('items', [])
    if not results:
        return ArticleInfo(
            status="error",
            content=ArticleInfoContent(),
            error=f"No metadata found for title: {title}"
        )
    return ArticleInfo(
        status = status,
        content = ArticleInfoContent(
            publisher = results[0]['publisher'],
            doi = results[0]['DOI'],
            source = results[0]['source'],
            title = results[0]['title'][0],
            author= results[0]['author'],
            year = results[0]['created']['date-parts'][0][0],
            citations = results[0]['is-referenced-by-count']
        )
    )
    # final_result = {
    #     "status": status,
    #     "results": [
    #         {
    #             'publisher': result['publisher'],
    #             'doi': result['DOI'],
    #             'source': result['source'],
    #             'title': result['title'][0],
    #             'author': result['author'],
    #             'year': result['created']['date-parts'][0][0]
    #         }
    #         for result in results
    #     ]
    # }    
    # 1. Input Validation (Defense against Local File Injection)
    # clean_title = sanitize_input(title)
    # if len(clean_title) < 5:
    #     return {"error": "Title too short to search."}
    # try:
    #     # 2. External API Call
    
    #     if response.status_code != 200:
    #         return {"error": f"API returned status {response.status_code}"}

    #     item = response.json().get("message", {})['items'][0] if response.json().get("message", {}).get('items') else {}
    #     print(response.json())
    #     # 3. Response Sanitization (Defense against Malicious API Response)
    #     return ArticleInfo(
    #         source = "crossref-api",
    #         trusted = False,
    #         official_title = sanitize_data(item.get("title", [""])[0]),
    #         publisher = sanitize_data(item.get("publisher", "")),
    #         year = item.get("created", {}).get("date-parts", [[None]])[0][0],
    #         type = item.get("type", ""),
    #         doi = item.get("DOI", ""),
    #         citations = item.get("is-referenced-by-count", 0),
    #         authors = item.get("author", []),            
    #     )

    # except Exception as e:
    #     return ArticleInfo(error=f'Connection Failed: {str(e)}')


if __name__ == "__main__":
    mcp.run(transport='http', port=8787)