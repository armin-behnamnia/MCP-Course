from fastmcp import FastMCP
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
from typing import Optional
import pymupdf4llm

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

class AccessDeniedError(Exception):
    """Raised when a file access rule is violated."""

mcp = FastMCP(
    name="PDFFileServer",
    instructions=(
        "A read-only PDF file server. "
        "Use `list_pdf_files` to discover files by keyword. "
        "Use `read_pdf` to load a PDF and receive its content as Markdown. "
        "Restricted files require a valid access token."
    ),
)

def _resolve_safe(base: Path, rel: str) -> Path:
    """
    Resolve *rel* relative to *base* and verify it stays inside *base*.
    Raises AccessDeniedError on any path-traversal attempt.
    """
    rel_clean = rel.lstrip("/\\")
    candidate = (base / rel_clean).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise AccessDeniedError(
            f"Path traversal detected: '{rel}' escapes the root directory."
        )
    return candidate


def _assert_pdf(path: Path) -> None:
    if path.suffix.lower() != ".pdf":
        raise AccessDeniedError(
            f"Only .pdf files are accessible. '{path.name}' is not allowed."
        )


def _assert_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path.name}'")

def _validate_token(provided: Optional[str]) -> None:
    if not RESTRICTED_TOKEN:
        raise AccessDeniedError(
            "RESTRICTED_TOKEN is not configured on this server. "
            "Restricted files are currently unavailable."
        )
    if provided != RESTRICTED_TOKEN:
        raise AccessDeniedError(
            "Invalid or missing token. Access to restricted files is denied."
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





# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_pdfs(directory: Path, keyword: str) -> list[str]:
    """
    Return sorted relative POSIX paths of .pdf files inside *directory*
    whose filename contains *keyword* (case-insensitive).
    Empty *keyword* matches everything.
    """
    if not directory.is_dir():
        return []
    kw = keyword.lower()
    return sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*.pdf")
        if kw in p.name.lower()
    )


def _pdf_to_markdown(path: Path) -> str:
    try:
        return pymupdf4llm.to_markdown(str(path))
    except Exception as e:
        logger.info(e)
        with open(str(path), 'rb') as f:
            return f.read()


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
    folder = folder.strip().lower()

    if folder == "allowed":
        path = _resolve_safe(ALLOWED_DIR, file_id)
        _assert_pdf(path)
        _assert_exists(path)

    elif folder == "restricted":
        _validate_token(token)
        path = _resolve_safe(RESTRICTED_DIR, file_id)
        _assert_pdf(path)
        _assert_exists(path)

    else:
        raise ValueError(
            f"Unknown folder '{folder}'. Must be 'allowed' or 'restricted'."
        )

    return _pdf_to_markdown(path)

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
