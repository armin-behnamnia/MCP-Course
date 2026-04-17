from fastmcp import FastMCP
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
from typing import Optional
from utils import _extract_section_content, _parse_bold_headers, _validate_token, _find_pdfs, _read_pdf_core, _generate_mini_summary
from fastmcp.prompts import PromptResult, Message
from openai import OpenAI

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

openai_client = OpenAI(
    base_url="http://localhost:8015/v1",
    api_key=""
)
MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

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
    description=(
        "Static configuration and capability metadata for this PDF server. "
        "Shows which directories are configured, whether a restricted token is "
        "set, and what operations are available. Safe to cache indefinitely — "
        "only changes when the server is restarted with different env vars."
    ),
    mime_type="application/json",
    tags={"config", "metadata"},
)
def resource_server_config() -> str:
    """
    Returns a JSON object describing the server's current configuration.

    Deliberately omits secret values — only reports whether they are set.
    Omits absolute paths — only reports whether directories exist.
    """
    return json.dumps({
        "server": "PDFFileServer",
        "folders": {
            "allowed": {
                "exists": ALLOWED_DIR.is_dir(),
            },
            "restricted": {
                "exists": RESTRICTED_DIR.is_dir(),
                "token_is_set": bool(RESTRICTED_TOKEN),
            },
        },
        "capabilities": {
            "list_by_keyword": True,
            "read_as_markdown": True,
            "write": False,
            "delete": False,
        },
        "allowed_file_types": [".pdf"],
    }, indent=2)


@mcp.resource(
    "stats://files",
    name="FileStats",
    description=(
        "Live file counts for each folder, computed fresh on every read. "
        "Unlike config://server this changes as PDFs are added or removed, "
        "making it a good candidate for client-side polling or subscription. "
        "Returns JSON with total counts, subfolder breakdown, and a timestamp."
    ),
    mime_type="application/json",
    tags={"stats", "metadata"},
)
def resource_file_stats() -> dict:
    """
    Returns a JSON object with current PDF counts per folder.

    Executed on every read so it always reflects the live filesystem state.
    This is a genuinely dynamic resource: it has a permanent URI but its
    content changes, which is exactly what resources/subscribe is designed for.
    """
    def _count_by_subdir(root: Path) -> dict:
        if not root.is_dir():
            return {}
        counts: dict[str, int] = {}
        for pdf in root.rglob("*.pdf"):
            subdir = pdf.parent.relative_to(root).as_posix()
            counts[subdir] = counts.get(subdir, 0) + 1
        return counts

    allowed_counts    = _count_by_subdir(ALLOWED_DIR)
    restricted_counts = _count_by_subdir(RESTRICTED_DIR)

    return json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "allowed": {
            "total": sum(allowed_counts.values()),
            "by_subfolder": allowed_counts,
        },
        "restricted": {
            "total": sum(restricted_counts.values()),
            "note": "file names are visible; content requires a token",
            "by_subfolder": restricted_counts,
        },
    }, indent=2)


@mcp.resource(
    "catalog://allowed",
    name="AllowedCatalog",
    description=(
        "Complete index of all publicly accessible PDF files. "
        "Returns a JSON array of objects, each with 'id' (pass directly to "
        "read_pdf), 'filename', 'size_bytes', and 'subfolder'. "
        "Recomputed on every read. Clients can cache this and refresh when "
        "stats://files shows a count change. Restricted file names are NOT "
        "included here — use list_pdf_files('') for a cross-folder listing."
    ),
    mime_type="application/json",
    tags={"catalog", "allowed"},
)
def resource_allowed_catalog() -> str:
    """
    Returns a JSON array describing every allowed PDF.

    This is a dynamic resource: permanent URI, but content updates as files
    are added/removed. Appropriate as a resource (not a tool) because:
      - It is non-parametric (no arguments needed)
      - It represents the server's own declared state
      - Clients benefit from caching it between requests
      - It has clear subscription semantics (notify when files change)
    """
    if not ALLOWED_DIR.is_dir():
        return json.dumps([])

    entries = []
    for pdf in sorted(ALLOWED_DIR.rglob("*.pdf")):
        rel    = pdf.relative_to(ALLOWED_DIR).as_posix()
        subdir = pdf.parent.relative_to(ALLOWED_DIR).as_posix()
        entries.append({
            "id":         rel,
            "filename":   pdf.name,
            "subfolder":  subdir,
            "size_bytes": pdf.stat().st_size,
        })

    return json.dumps(entries, indent=2)




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
    name="compare_sections",
    description=(
        "Compare a specific section across two documents. "
        "Returns differences, similarities, and proper citations."
    ),
    tags={"comparison", "analysis"},
)
def prompt_compare_sections(
    doc1: str,
    doc2: str,
    section: str,
) -> str:
    """
    Compare the same section across two documents.
    
    Parameters
    ----------
    doc1, doc2 : str
        Document identifiers (as returned by list_pdf_files)
    section : str
        Section name to compare
    """
    return (
        f"Please compare the '{section}' section between:\n"
        f"- Document 1: {doc1}\n"
        f"- Document 2: {doc2}\n\n"
        f"Steps:\n"
        f"1. Use extract_section(file_id='{doc1}', header='{section}', folder='allowed')\n"
        f"2. Use extract_section(file_id='{doc2}', header='{section}', folder='allowed')\n"
        f"3. Summarize key points from each\n"
        f"4. Note similarities and differences\n"
        f"5. Cite as [Document: X, Section: {section}]\n\n"
        f"If either document doesn't have this section, use extract_headers to suggest alternatives."
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
    name="extract_with_citations",
    description=(
        "Extract specific information with academic-style citations including direct quotes. "
        "Enforces rigorous citation discipline."
    ),
    tags={"citation", "academic"},
)
def prompt_extract_with_citations(
    research_question: str,
) -> PromptResult:
    """
    Academic research extraction with strict citation requirements.
    
    Parameters
    ----------
    research_question : str
        The specific question to answer from the documents
    """
    system_msg = (
        "You are an academic research assistant. Citation requirements:\n\n"
        "EVERY factual claim must include:\n"
        "- Source document name\n"
        "- Section name\n"
        "- A brief direct quote (1-2 sentences maximum)\n\n"
        "Format: [Document: filename.pdf, Section: 'Header'] Quote: \"...\"\n\n"
        "Rules:\n"
        "- Use extract_section to get precise content\n"
        "- Never paraphrase without showing the original quote\n"
        "- If sources conflict, cite both and note the discrepancy\n"
        "- If information is not in the documents, state this explicitly"
    )
    
    user_msg = (
        f"Research Question: {research_question}\n\n"
        f"Please:\n"
        f"1. Identify relevant documents with list_pdf_files\n"
        f"2. Use extract_headers to locate relevant sections\n"
        f"3. Use extract_section to retrieve content\n"
        f"4. Answer the question with full citations including quotes\n"
        f"5. If sources disagree, present both views with citations"
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

