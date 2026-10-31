"""File-system tools for resume reading, writing, listing, and searching.

Each public function returns structured data so it can be used directly as an
LLM tool response.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_READ_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _error(message: str, filepath: str | Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"success": False, "error": message}
    if filepath is not None:
        payload["filepath"] = str(filepath)
    return payload


def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the 'pypdf' package.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {page_number}]\n{text}".strip())
    return "\n\n".join(page for page in pages if page)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("DOCX support requires the 'python-docx' package.") from exc

    document = Document(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]

    table_rows: list[str] = []
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            table_rows.append(" | ".join(cell for cell in cells if cell))

    parts = [text for text in paragraphs + table_rows if text.strip()]
    return "\n".join(parts)


def read_file(filepath: str) -> dict[str, Any]:
    """Read a TXT, PDF, or DOCX resume and return content plus metadata."""

    path = Path(filepath).expanduser()
    try:
        if not path.exists():
            return _error("File does not exist.", filepath)
        if not path.is_file():
            return _error("Path is not a file.", filepath)

        extension = path.suffix.lower()
        if extension not in SUPPORTED_READ_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_READ_EXTENSIONS))
            return _error(f"Unsupported file type '{extension}'. Supported: {supported}.", filepath)

        if extension == ".txt":
            content = _read_txt(path)
        elif extension == ".pdf":
            content = _read_pdf(path)
        else:
            content = _read_docx(path)

        return {
            "success": True,
            "content": content,
            "metadata": {
                **_metadata(path),
                "character_count": len(content),
                "word_count": len(content.split()),
            },
        }
    except Exception as exc:  # Keep tool responses graceful for LLM callers.
        return _error(str(exc), filepath)


def list_files(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
    """List files in a directory, optionally filtering by extension."""

    path = Path(directory).expanduser()
    if extension:
        extension = extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"

    if not path.exists():
        return [_error("Directory does not exist.", directory)]
    if not path.is_dir():
        return [_error("Path is not a directory.", directory)]

    files: list[dict[str, Any]] = []
    try:
        for item in sorted(path.iterdir(), key=lambda entry: entry.name.lower()):
            if not item.is_file():
                continue
            if extension and item.suffix.lower() != extension:
                continue
            files.append({"success": True, **_metadata(item)})
        return files
    except Exception as exc:
        return [_error(str(exc), directory)]


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """Write text content to a file, creating parent directories as needed."""

    path = Path(filepath).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "message": "File written successfully.",
            "metadata": _metadata(path),
        }
    except Exception as exc:
        return _error(str(exc), filepath)


def search_in_file(filepath: str, keyword: str) -> dict[str, Any]:
    """Case-insensitive keyword search with surrounding text context."""

    if not keyword:
        return _error("Keyword must not be empty.", filepath)

    result = read_file(filepath)
    if not result.get("success"):
        return result

    content = result["content"]
    content_lower = content.lower()
    keyword_lower = keyword.lower()
    matches: list[dict[str, Any]] = []
    start = 0
    context_chars = 80

    while True:
        index = content_lower.find(keyword_lower, start)
        if index == -1:
            break
        context_start = max(0, index - context_chars)
        context_end = min(len(content), index + len(keyword) + context_chars)
        matches.append(
            {
                "match_number": len(matches) + 1,
                "start_index": index,
                "end_index": index + len(keyword),
                "matched_text": content[index : index + len(keyword)],
                "context": content[context_start:context_end].replace("\n", " "),
            }
        )
        start = index + len(keyword)

    return {
        "success": True,
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
        "metadata": result["metadata"],
    }
