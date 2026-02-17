from fastmcp import FastMCP
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ALLOWED_DIR: Path = Path(os.environ.get("ALLOWED_DIR", "")).resolve()
RESTRICTED_DIR: Path = Path(os.environ.get("RESTRICTED_DIR", "")).resolve()
RESTRICTED_TOKEN: str = os.environ.get("RESTRICTED_TOKEN", "")

class AccessDeniedError(Exception):
    """Raised when a file access rule is violated."""

def _resolve_safe(base: Path, rel: str) -> Path:
    """
    Resolve *rel* relative to *base* and ensure the result stays within *base*.
    Raises AccessDeniedError on path-traversal attempts.
    """
    # Normalise: strip leading slashes so Path(base) / "/etc/passwd" won't escape
    rel_clean = rel.lstrip("/\\").replace("..", "")
    candidate = (base / rel_clean).resolve()
    try:
        candidate.relative_to(base)  # raises ValueError if not a sub-path
    except ValueError:
        raise AccessDeniedError(
            f"Path traversal detected: '{rel}' escapes the root directory."
        )
    return candidate


def _assert_pdf(path: Path) -> None:
    """Raise AccessDeniedError if the file is not a .pdf."""
    if path.suffix.lower() != ".pdf":
        raise AccessDeniedError(
            f"Only .pdf files are accessible. '{path.name}' is not allowed."
        )


def _assert_exists(path: Path) -> None:
    """Raise FileNotFoundError with a safe message (no full path leak)."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path.name}'")

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
    "pdf://allowed/{rel_path}",
    name="AllowedPDF",
    description=(
        "Returns the raw bytes of a PDF from the publicly accessible folder. "
        "rel_path is the file's relative path within the allowed directory "
        "(e.g. 'reports/annual_2024.pdf')."
    ),
    mime_type="application/pdf",
    tags={"pdf", "allowed"},
)
def resource_allowed_pdf(rel_path: str) -> bytes:
    path = _resolve_safe(ALLOWED_DIR, rel_path)
    _assert_pdf(path)
    _assert_exists(path)
    return path.read_bytes()



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