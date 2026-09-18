"""
Document metadata extraction agent using PydanticAI.

Reads a PDF (as raw bytes, sent inline to a multimodal model) and returns
a typed BookMetadata object. No manual PDF-text parsing needed — Claude/
Gemini/GPT read the PDF directly and PydanticAI validates + retries the
structured output against the schema.

    uv add pydantic-ai

Requires an API key for whichever model you pick, e.g.:
    export ANTHROPIC_API_KEY=...
"""

import argparse
import glob
import json
import os
import sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pypdf import PdfReader, PdfWriter

load_dotenv(Path(__file__).with_name(".env"))

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
DEFAULT_MAX_PAGES = 10


def _max_pages() -> int:
    raw = os.getenv("PDF_METADATA_MAX_PAGES", str(DEFAULT_MAX_PAGES))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"PDF_METADATA_MAX_PAGES must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(
            f"PDF_METADATA_MAX_PAGES must be a positive integer, got {raw!r}"
        )
    return value


def _pdf_payload(pdf_bytes: bytes) -> tuple[bytes, int]:
    """Return the first N pages of a PDF plus its total page count.

    Only the front matter is sent to the model; bibliographic metadata lives
    on the title and copyright pages.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    writer = PdfWriter()
    for page in reader.pages[: _max_pages()]:
        writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue(), total_pages


def _front_matter_text(pdf_bytes: bytes) -> str:
    """Extract readable text from the first N pages of a PDF.

    Text-only chat models (such as DeepSeek via CheaperInference) cannot
    accept PDF file input, so they receive this text instead of raw bytes.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    parts = []
    for index, page in enumerate(reader.pages[: _max_pages()], start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = text.strip()
        if cleaned:
            parts.append(f"--- page {index} ---\n{cleaned}")
    return "\n\n".join(parts)


def _model_input(pdf_bytes: bytes, send_bytes: bytes) -> list:
    """Build the agent prompt for the configured model.

    CheaperInference text models get extracted front-matter text (small
    enough for the proxy's request limits); all other models get the raw
    PDF bytes as before.
    """
    if os.getenv("PDF_METADATA_MODEL", DEFAULT_MODEL).startswith("cheaper:"):
        text = _front_matter_text(pdf_bytes)
        if not text.strip():
            raise RuntimeError(
                "cheaper: models receive extracted PDF text, but this PDF "
                "has no extractable text (likely scanned images). OCR the "
                "PDF or use a multimodal provider instead."
            )
        return [
            "Extract full bibliographic metadata from this document's "
            "front matter text.",
            text,
        ]
    return [
        "Extract full bibliographic metadata from this document.",
        BinaryContent(data=send_bytes, media_type="application/pdf"),
    ]


def _provider_and_model() -> tuple[str | None, str | None]:
    """Split PDF_METADATA_MODEL into (provider, model) for recording on output."""
    raw = os.getenv("PDF_METADATA_MODEL", DEFAULT_MODEL)
    if ":" in raw:
        provider, name = raw.split(":", 1)
        return provider or None, name or None
    return None, raw or None


def _build_model():
    """Resolve the model for the extraction agent.

    Returns the ``PDF_METADATA_MODEL`` string unchanged, except for the
    ``meta:`` and ``cheaper:`` schemes, which build an OpenAI-compatible
    chat model pointed at Meta's API or CheaperInference. Those endpoints
    only implement the Chat Completions API and only support
    ``tool_choice="auto"``, so the model is constructed explicitly with a
    profile that disables forced tool use (PydanticAI's ``openai:`` prefix
    would use the Responses API instead).
    """
    model_name = os.getenv("PDF_METADATA_MODEL", DEFAULT_MODEL)
    if model_name.startswith("meta:"):
        scheme, key_var, url_var, default_url = (
            "meta",
            "META_API_KEY",
            "META_BASE_URL",
            "https://api.meta.ai/v1",
        )
    elif model_name.startswith("cheaper:"):
        scheme, key_var, url_var, default_url = (
            "cheaper",
            "CHEAPER_INFERENCE_API_KEY",
            "CHEAPER_INFERENCE_BASE_URL",
            "https://api.cheaperinference.com/v1",
        )
    else:
        return model_name

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.getenv(key_var)
    if not api_key:
        raise RuntimeError(
            f"PDF_METADATA_MODEL uses the '{scheme}:' scheme but {key_var} is not set"
        )
    _, name = model_name.split(":", 1)
    if not name.strip():
        raise RuntimeError(
            f"PDF_METADATA_MODEL uses the '{scheme}:' scheme but names no model"
        )
    provider = OpenAIProvider(
        base_url=os.getenv(url_var, default_url),
        api_key=api_key,
    )
    return OpenAIChatModel(
        name,
        provider=provider,
        profile=OpenAIModelProfile(openai_supports_tool_choice_required=False),
    )


class ISBNEntry(BaseModel):
    value: str = Field(description="ISBN-10 or ISBN-13, digits only")
    format: str | None = Field(
        default=None, description="Labeled binding or format, e.g. cloth, ePDF, EPUB"
    )


class BookMetadata(BaseModel):
    title: str
    subtitle: str | None = None
    authors: list[str] = Field(default_factory=list)
    isbns: list[ISBNEntry] = Field(default_factory=list)
    publisher: str | None = None
    publication_date: str | None = Field(
        default=None, description="Best available date, e.g. '2023' or '2023-05'"
    )
    edition: str | None = None
    language: str | None = None
    page_count: int | None = None
    subjects: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None
    document_type: str | None = Field(
        default=None, description="e.g. book, article, white paper, magazine"
    )
    file_size_bytes: int | None = Field(default=None, ge=0)
    original_filename: str | None = None
    run_time_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provider: str | None = Field(
        default=None,
        description="Provider that performed the extraction, e.g. 'anthropic'",
    )
    model: str | None = Field(
        default=None,
        description="Model that performed the extraction, e.g. 'claude-sonnet-4-6'",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Agent's own confidence in this extraction"
    )
    suggested_filename: str = Field(
        description=(
            "Standardized filename (no extension) built from this metadata, "
            "e.g. 'Author_Lastname_-_Title'"
        )
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    subtitle TEXT,
    authors TEXT NOT NULL DEFAULT '[]',
    isbns TEXT NOT NULL DEFAULT '[]',
    publisher TEXT,
    publication_date TEXT,
    edition TEXT,
    language TEXT,
    page_count INTEGER,
    subjects TEXT NOT NULL DEFAULT '[]',
    keywords TEXT NOT NULL DEFAULT '[]',
    summary TEXT,
    document_type TEXT,
    file_size_bytes INTEGER,
    original_filename TEXT,
    run_time_seconds REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    provider TEXT,
    model TEXT,
    confidence REAL NOT NULL,
    suggested_filename TEXT NOT NULL,
    extracted_at TEXT NOT NULL
)
"""

_JSON_COLUMNS = ("authors", "isbns", "subjects", "keywords")


def _db_path() -> Path:
    return Path(os.getenv("PDF_METADATA_DB", Path(__file__).with_name("metadata.db")))


def save_metadata(meta: BookMetadata, db_path: Path | None = None) -> int:
    """Insert extracted metadata into the SQLite database; return the row id."""
    path = db_path if db_path is not None else _db_path()
    record = meta.model_dump()
    for column in _JSON_COLUMNS:
        record[column] = json.dumps(record[column])
    record["extracted_at"] = datetime.now(timezone.utc).isoformat()
    columns = ", ".join(record)
    placeholders = ", ".join(["?"] * len(record))
    with sqlite3.connect(path) as conn:
        conn.execute(_SCHEMA)
        cursor = conn.execute(
            f"INSERT INTO metadata ({columns}) VALUES ({placeholders})",
            list(record.values()),
        )
        return cursor.lastrowid or 0


def list_metadata(db_path: Path | None = None) -> list[dict]:
    """Return all saved metadata records, oldest first."""
    path = db_path if db_path is not None else _db_path()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(_SCHEMA)
        rows = conn.execute("SELECT * FROM metadata ORDER BY id").fetchall()
    records = []
    for row in rows:
        record = dict(row)
        for column in _JSON_COLUMNS:
            record[column] = json.loads(record[column])
        records.append(record)
    return records


extraction_agent = Agent(
    model=_build_model(),
    output_type=BookMetadata,
    system_prompt=(
        "You extract bibliographic metadata from book/document PDFs. "
        "Look at the title page, copyright page, and cover for ISBNs, "
        "author names, publisher, and edition. Pair each ISBN with its labeled "
        "binding or format (such as cloth, ePDF, or EPUB); leave the format "
        "null if it is not stated. Extract subjects and keywords when present, "
        "write a brief summary when the content supports one, and classify "
        "the document type (such as book, article, white paper, or magazine). "
        "Leave unknown bibliographic facts null rather than guessing. "
        "Use empty lists for missing subjects or keywords. "
        "Normalize ISBNs to digits only (strip dashes/spaces). "
        "Original filename, file size, run time, token counts, page count, "
        "provider, and model are set by the application.\n\n"
        "Also produce `suggested_filename`, a standardized filename (no "
        "extension) in the form 'Lastname_-_Title'. Rules:\n"
        "- Use the first author's last name only; append 'et_al' if there "
        "are 3+ authors, or 'and_Lastname2' if there are exactly 2.\n"
        "- Replace spaces in the title with underscores; strip characters "
        'that are illegal in Windows/Linux filenames (\\ / : * ? " < > |).\n'
        "- Do not include the publication year or any date in the filename.\n"
        "- If authors are unknown, start the filename with the title."
    ),
)


def _rename_pdf(pdf_path: Path, suggested_filename: str) -> None:
    if (
        not suggested_filename
        or suggested_filename in {".", ".."}
        or any(char in suggested_filename for char in ("/", "\\", "\0"))
    ):
        raise ValueError("Suggested filename must be a single nonempty basename")

    destination = pdf_path.with_name(f"{suggested_filename}{pdf_path.suffix}")
    if destination == pdf_path:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    pdf_path.rename(destination)


def extract_metadata(pdf_path: Path, *, rename: bool = False) -> BookMetadata:
    """Run the agent once on a single PDF and return validated metadata."""
    pdf_bytes = pdf_path.read_bytes()
    send_bytes, total_pages = _pdf_payload(pdf_bytes)
    start = perf_counter()
    result = extraction_agent.run_sync(_model_input(pdf_bytes, send_bytes))
    elapsed = perf_counter() - start
    result.output.original_filename = pdf_path.name
    result.output.file_size_bytes = len(pdf_bytes)
    result.output.run_time_seconds = elapsed
    result.output.input_tokens = result.usage.input_tokens
    result.output.output_tokens = result.usage.output_tokens
    result.output.total_tokens = result.usage.total_tokens
    result.output.page_count = total_pages
    result.output.provider, result.output.model = _provider_and_model()
    save_metadata(result.output)
    if rename:
        _rename_pdf(pdf_path, result.output.suggested_filename)
    return result.output  # already validated as BookMetadata


async def extract_metadata_async(
    pdf_path: Path, *, rename: bool = False
) -> BookMetadata:
    """Async version — use this if you're calling it from FastAPI."""
    pdf_bytes = pdf_path.read_bytes()
    send_bytes, total_pages = _pdf_payload(pdf_bytes)
    start = perf_counter()
    result = await extraction_agent.run(_model_input(pdf_bytes, send_bytes))
    elapsed = perf_counter() - start
    result.output.original_filename = pdf_path.name
    result.output.file_size_bytes = len(pdf_bytes)
    result.output.run_time_seconds = elapsed
    result.output.input_tokens = result.usage.input_tokens
    result.output.output_tokens = result.usage.output_tokens
    result.output.total_tokens = result.usage.total_tokens
    result.output.page_count = total_pages
    result.output.provider, result.output.model = _provider_and_model()
    save_metadata(result.output)
    if rename:
        _rename_pdf(pdf_path, result.output.suggested_filename)
    return result.output


def _expand_paths(patterns: list[Path]) -> list[Path]:
    """Expand glob patterns to PDF paths, preserving order without duplicates.

    Plain paths pass through untouched (missing files fail at extraction);
    patterns matching only non-files, or nothing, contribute no paths.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        text = str(pattern)
        if glob.has_magic(text):
            matches = [
                Path(m)
                for m in sorted(glob.glob(text, recursive=True))
                if Path(m).is_file()
            ]
        else:
            matches = [pattern]
        for path in matches:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metadata from PDFs")
    parser.add_argument(
        "pdf_path",
        type=Path,
        nargs="*",
        help="PDF file(s) or glob pattern(s), e.g. 'pdfs/*.pdf'",
    )
    parser.add_argument(
        "--rename", action="store_true", help="Rename the PDF to its suggested filename"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List metadata saved in the SQLite database",
    )
    args = parser.parse_args()
    if args.list:
        print(json.dumps(list_metadata(), indent=2))
        return
    if not args.pdf_path:
        parser.error("pdf_path is required unless --list is given")
    paths = _expand_paths(args.pdf_path)
    if not paths:
        parser.error(f"no files match: {' '.join(str(p) for p in args.pdf_path)}")
    if len(paths) == 1:
        meta = extract_metadata(paths[0], rename=args.rename)
        print(meta.model_dump_json(indent=2))
    else:
        results = [extract_metadata(path, rename=args.rename) for path in paths]
        print(json.dumps([meta.model_dump() for meta in results], indent=2))


if __name__ == "__main__":
    main()
