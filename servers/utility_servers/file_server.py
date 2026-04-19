from fastmcp import FastMCP
from .utils import _validate_token, _find_pdfs, _read_pdf_core, _parse_bold_headers, _extract_section_content, _generate_mini_summary
from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Optional

PROXY = "http://192.168.10.2:3129" 

load_dotenv()

ALLOWED_DIR: Path = Path(os.environ.get("ALLOWED_DIR", "")).resolve()
RESTRICTED_DIR: Path = Path(os.environ.get("RESTRICTED_DIR", "")).resolve()
RESTRICTED_TOKEN: str = os.environ.get("RESTRICTED_TOKEN", "")

mcp = FastMCP(
    name="File MCP Server",
    instructions=(
        "File MCP Server: Provides tools for searching and reading PDF files stored on the server. "
        "Use list_pdf_files to find PDFs by keyword, then read_pdf to get their content in Markdown format. "
        "For structured access, use extract_headers to get section headers and extract_section to retrieve specific sections. "
        "Some PDFs are in a restricted folder and require a valid token for access."
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
    tags={"pdf", "search", "requires_token"},
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
    tags={"pdf", "read", "requires_token"},
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
    tags={"pdf", "headers", "requires_token"},
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
    tags={"pdf", "section", "requires_token"},
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


@mcp.tool(tags={"context_reshaping", "requires_token"})
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

if __name__ == "__main__":
    mcp.run(transport='http', port=8030)