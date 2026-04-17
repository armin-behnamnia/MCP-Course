from fastmcp import FastMCP
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
from typing import Optional
from .utils import _extract_section_content, _parse_bold_headers, _validate_token, _find_pdfs, _read_pdf_core, _generate_mini_summary
from fastmcp.prompts import PromptResult, Message
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
import shutil

from .rag_service import search_papers, initialize_index

# Logging setup
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()

ALLOWED_DIR: Path = Path(os.environ.get("ALLOWED_DIR", "")).resolve()
RESTRICTED_DIR: Path = Path(os.environ.get("RESTRICTED_DIR", "")).resolve()
RESTRICTED_TOKEN: str = os.environ.get("RESTRICTED_TOKEN", "")
RAG_DIR: str = Path(os.environ.get("RAG_DIR", "")).resolve()
RAG_DB_DIR: str = Path(__file__).parent / "./chroma_db"
LLM_BASE_URL = "http://localhost:11434/v1"

print(RAG_DB_DIR, RAG_DIR)
openai_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=""
)
MODEL_NAME="qwen3:0.6b"

mcp = FastMCP(
    name="PDFFileServer",
    instructions=(
        "A read-only PDF file server. "
        "Use `list_pdf_files` to discover files by keyword. "
        "Use `read_pdf` to load a PDF and receive its content as Markdown. "
        "Restricted files require a valid access token."
    ),
)

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
db_scheduler.add_job(scheduled_indexing, kwargs={"db_object": rag_db_object}, trigger='interval', minutes=15, id='lraa_sync_job')
db_scheduler.start()

@mcp.prompt(
    name="find_and_summarize",
    description=(
        "Find documents by keyword and summarize a specific section. "
        "Ideal for targeted research queries."
    ),
    tags={"research", "summary"},
)
def prompt_find_and_summarize(
    keyword: str,
    section: str,
) -> PromptResult:
    """
    Guide the LLM to find documents and extract a specific section.
    
    Parameters
    ----------
    keyword : str
        Search term for finding relevant documents
    section : str
        Name of the section to extract (e.g., 'Methods', 'Results')
    """
    system_msg = (
        "You are a research assistant with access to PDF documents. "
        "Always provide citations in the format [Document: X, Section: Y]. "
        "Extract only the requested section - never load full documents."
    )
    
    user_msg = (
        f"Please find and summarize the '{section}' section from documents about '{keyword}':\n\n"
        f"Steps:\n"
        f"1. Use list_pdf_files(keyword='{keyword}') to find relevant documents\n"
        f"2. For each document, use extract_headers to verify '{section}' exists\n"
        f"3. Use extract_section to get the content\n"
        f"4. Provide a summary with citation: [Document: X, Section: {section}]\n"
        f"5. If no documents have this section, list available sections"
    )
    
    return PromptResult(
        messages=[
            Message(role="assistant", content=system_msg),
            Message(role="user", content=user_msg),
        ]
    )


@mcp.prompt(
    name="research_topic",
    description=(
        "Comprehensive research workflow: find documents, extract relevant sections, "
        "synthesize findings with citations. Best for open-ended research questions."
    ),
    tags={"research", "comprehensive"},
)
def prompt_research_topic(
    topic: str,
    max_documents: int = 5,
) -> PromptResult:
    """
    Deep research workflow with proper citation discipline.
    
    Parameters
    ----------
    topic : str
        Research topic or question
    max_documents : int
        Maximum number of documents to analyze (default: 5)
    """
    system_msg = (
        "You are a thorough research assistant. Your responses must:\n"
        "1. Include citations for every factual claim: [Document: X, Section: Y]\n"
        "2. Extract only relevant sections, never full documents\n"
        "3. Note contradictions between sources\n"
        "4. Acknowledge gaps in available information\n"
        "5. Use extract_headers before extract_section to verify structure"
    )
    
    user_msg = (
        f"Research topic: {topic}\n\n"
        f"Please conduct a comprehensive analysis:\n\n"
        f"1. Use list_pdf_files to find up to {max_documents} relevant documents\n"
        f"2. For each document:\n"
        f"   a. Use extract_headers to see structure\n"
        f"   b. Identify relevant sections\n"
        f"   c. Use extract_section to retrieve content\n"
        f"3. Synthesize findings with proper citations\n"
        f"4. Note any contradictions or information gaps\n"
        f"5. Provide a conclusion\n\n"
        f"If you need clarification about which sections to focus on, ask me."
    )
    
    return PromptResult(
        messages=[
            Message(role="assistant", content=system_msg),
            Message(role="user", content=user_msg),
        ]
    )


@mcp.prompt(
    name="restricted_document_access",
    description=(
        "Guided workflow for accessing token-gated restricted documents. "
        "Handles authentication and section extraction."
    ),
    tags={"restricted", "authentication"},
)
def prompt_restricted_document_access(
    document_name: str,
) -> PromptResult:
    """
    Guide for accessing restricted documents with proper token handling.
    
    Parameters
    ----------
    document_name : str
        Name or partial name of the restricted document
    """
    system_msg = (
        "You are helping access restricted documents. Important rules:\n"
        "1. ALWAYS ask the user for the access token - never guess\n"
        "2. Explain that the token is required for restricted files\n"
        "3. Only attempt access after receiving a token from the user\n"
        "4. If the token is invalid, inform the user clearly"
    )
    
    user_msg = (
        f"I need to access: {document_name}\n\n"
        f"Please follow this workflow:\n"
        f"1. Use list_pdf_files(keyword='{document_name}') to find the document\n"
        f"2. Check if it's in the 'restricted' folder\n"
        f"3. If restricted, ask me: 'This document requires an access token. Please provide it.'\n"
        f"4. Once I give you the token, use extract_headers with the token parameter\n"
        f"5. Show me the available sections\n"
        f"6. Wait for me to specify which section I need\n"
        f"7. Then use extract_section with the token to retrieve it"
    )
    
    return PromptResult(
        messages=[
            Message(role="assistant", content=system_msg),
            Message(role="user", content=user_msg),
        ]
    )



# ===========================================================================
# TOOLS — caller-driven, parametric operations
#
# Tools are the right abstraction for everything that requires a runtime
# parameter (keyword, file_id, folder, token).  The LLM decides what to
# fetch; the server executes it.  Token-gating is only possible via tools
# because resources cannot receive caller-supplied arguments beyond the URI.
# ===========================================================================

@mcp.tool(
    name="list_pdf_files",
    description=(
        "Search for PDF files by keyword in their filename (case-insensitive). "
        "Searches both the allowed and restricted folders. "
        "Returns a list of objects each with 'id', 'folder', and 'filename'. "
        "Pass an empty string to list all PDF files across both folders. "
        "Use the returned 'id' and 'folder' values with read_pdf."
    ),
    tags={"pdf", "search", "context_reshaping"},
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
        
        section_text = _extract_section_content(markdown, section_target)
        
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

@mcp.tool(tags={'rag'}, name='search_research_papers')
def search_research_papers(query: str, n: int = 3) -> str:
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

        # Building a structured response for the LLM's context window
        formatted_output = [f"Found {len(results['results'])} relevant segments:\n"]
        
        for i, res in enumerate(results["results"], 1):
            formatted_output.append(
                {
                    "file": res.get('source', 'Unknown'),
                    "page": res.get('page', 'N/A'),
                    "content": res.get('content', ''),
                    "score": res.get('score', 0)
                }
            )
        print(formatted_output)
        return json.dumps(formatted_output)
        
    except Exception as e:
        return f"Error accessing the research repository: {str(e)}"

if __name__ == "__main__":

    # Quick self-check: warn if env vars are missing
    missing = []
    if not os.environ.get("ALLOWED_DIR"):
        missing.append("ALLOWED_DIR")
    if not os.environ.get("RESTRICTED_DIR"):
        missing.append("RESTRICTED_DIR")
    if not os.environ.get("RESTRICTED_TOKEN"):
        missing.append("RESTRICTED_TOKEN  (restricted files will be inaccessible)")

    if missing:
        print("⚠  Warning — the following environment variables are not set:")
        for m in missing:
            print(m)
        print()

    print("Starting PDF MCP File Server …")
    mcp.run(transport='http', port=8787)

