from fastmcp import FastMCP
from utils import _validate_token, _find_pdfs, _read_pdf_core
from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Optional


load_dotenv()

ALLOWED_DIR: Path = Path(os.environ.get("ALLOWED_DIR", "")).resolve()
RESTRICTED_DIR: Path = Path(os.environ.get("RESTRICTED_DIR", "")).resolve()
RESTRICTED_TOKEN: str = os.environ.get("RESTRICTED_TOKEN", "")


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
    return "{'name': 'pdfserver'}"

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


if __name__ == "__main__":
    mcp.run(transport='http', port=8787)