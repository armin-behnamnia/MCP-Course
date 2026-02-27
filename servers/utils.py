from pathlib import Path
import os
from typing import Optional
import re
from dotenv import load_dotenv
import pymupdf4llm

load_dotenv()

# Logging setup
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

class AccessDeniedError(Exception):
    """Raised when a file access rule is violated."""

ALLOWED_DIR: Path = Path(os.environ.get("ALLOWED_DIR", "")).resolve()
RESTRICTED_DIR: Path = Path(os.environ.get("RESTRICTED_DIR", "")).resolve()
RESTRICTED_TOKEN: str = os.environ.get("RESTRICTED_TOKEN", "")
BOLD_HEADER_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")


def _pdf_to_markdown(path: Path) -> str:
    try:
        return pymupdf4llm.to_markdown(str(path))
    except Exception as e:
        logger.info(e)
        with open(str(path), 'rb') as f:
            return f.read()

def _extract_title(file_id: str) -> str:
    path = _resolve_safe(ALLOWED_DIR, file_id)
    md_str = _pdf_to_markdown(path)
    search_result = re.search(r'##.+\n\n', md_str)
    if search_result:
        title = re.sub(r'[^A-Za-z0-9 ]+', '', search_result.group(0).strip())
        return title
    else:
        return path.split("/")[-1].split(" ")[-1].split(".")[0]

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

def _normalize_header(text: str):
    text = text.replace("*", ' ')
    text = re.sub(r'[ ][ ]+', ' ', re.sub(r'[^A-Za-z]', ' ', text)).strip().lower()
    return text

def _parse_bold_headers(markdown: str) -> list[str]:
    """
    Return an ordered list of headers found in *markdown*.
    A header is any line whose entire content (ignoring surrounding
    whitespace) is wrapped in double stars: **Header Text**
    """
    headers = []
    for line in markdown.splitlines():
        m = BOLD_HEADER_RE.match(line)
        if m:
            text = m.group(1)
            headers.append(_normalize_header(text))
    return headers


def _extract_section_content(markdown: str, header: str) -> str:
    """
    Return the block of text that follows *header* up to the next bold
    header or end of document.

    Matching is case-insensitive and strips surrounding whitespace from
    both the target header and the candidates found in the document.

    Raises ValueError if no matching header is found.
    """
    target = header.strip().lower()
    lines  = markdown.splitlines()

    start_idx: int | None = None
    for i, line in enumerate(lines):
        m = BOLD_HEADER_RE.match(line)
        if m:
            text = m.group(1)
            if _normalize_header(text) == target:
                start_idx = i + 1   # content begins on the line after the header
                break

    if start_idx is None:
        raise ValueError(
            f"Header '{header}' not found in the document. "
            "Use extract_headers to see available headers."
        )

    # Collect lines until the next bold header or EOF
    section_lines = []
    for line in lines[start_idx:]:
        if BOLD_HEADER_RE.match(line):
            break
        section_lines.append(line)

    return "\n".join(section_lines).strip()

def _read_pdf_core(file_id: str, folder: str, token: Optional[str] = None) -> str:
    """
    Core PDF reading logic shared by read_pdf, extract_headers, and extract_section.
    
    This is a plain Python function (not decorated with @mcp.tool) so it can be
    called directly by other tools without going through the MCP protocol layer.
    
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

def _rough_tokens(text: str) -> int:
    """Rough 4-chars-per-token heuristic."""
    return max(1, len(text) // 4)

def _generate_mini_summary(text: str) -> str:
    text = text.strip()
    # Logic to trim/summarize text to ~20 words
    if len(text) <= 203:
        return text
    return text[:100].strip() + "..." + text[-100:].strip()