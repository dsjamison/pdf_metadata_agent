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
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

load_dotenv(Path(__file__).with_name(".env"))


def _build_model():
    """Resolve the model for the extraction agent.

    Returns the ``PDF_METADATA_MODEL`` string unchanged, except for the
    ``meta:`` scheme, which builds an OpenAI-compatible chat model pointed
    at Meta's API. Meta only implements the Chat Completions API and only
    supports ``tool_choice="auto"``, so the model is constructed explicitly
    with a profile that disables forced tool use (PydanticAI's ``openai:``
    prefix would use the Responses API instead).
    """
    model_name = os.getenv("PDF_METADATA_MODEL", "anthropic:claude-sonnet-4-6")
    if not model_name.startswith("meta:"):
        return model_name

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.getenv("META_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PDF_METADATA_MODEL uses the 'meta:' scheme but META_API_KEY is not set"
        )
    _, name = model_name.split(":", 1)
    if not name.strip():
        raise RuntimeError(
            "PDF_METADATA_MODEL uses the 'meta:' scheme but names no model"
        )
    provider = OpenAIProvider(
        base_url=os.getenv("META_BASE_URL", "https://api.meta.ai/v1"),
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
    confidence: float = Field(
        ge=0.0, le=1.0, description="Agent's own confidence in this extraction"
    )
    suggested_filename: str = Field(
        description=(
            "Standardized filename (no extension) built from this metadata, "
            "e.g. 'Author_Lastname_-_Title_(2023)'"
        )
    )


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
        "Original filename, file size, run time, and token counts are set by "
        "the application.\n\n"
        "Also produce `suggested_filename`, a standardized filename (no "
        "extension) in the form 'Lastname_-_Title_(Year)'. Rules:\n"
        "- Use the first author's last name only; append 'et_al' if there "
        "are 3+ authors, or 'and_Lastname2' if there are exactly 2.\n"
        "- Replace spaces in the title with underscores; strip characters "
        'that are illegal in Windows/Linux filenames (\\ / : * ? " < > |).\n'
        "- Wrap the publication year in parentheses if known; omit the "
        "year segment entirely if unknown.\n"
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
    start = perf_counter()
    result = extraction_agent.run_sync(
        [
            "Extract full bibliographic metadata from this document.",
            BinaryContent(data=pdf_bytes, media_type="application/pdf"),
        ]
    )
    elapsed = perf_counter() - start
    result.output.original_filename = pdf_path.name
    result.output.file_size_bytes = len(pdf_bytes)
    result.output.run_time_seconds = elapsed
    result.output.input_tokens = result.usage.input_tokens
    result.output.output_tokens = result.usage.output_tokens
    result.output.total_tokens = result.usage.total_tokens
    if rename:
        _rename_pdf(pdf_path, result.output.suggested_filename)
    return result.output  # already validated as BookMetadata


async def extract_metadata_async(
    pdf_path: Path, *, rename: bool = False
) -> BookMetadata:
    """Async version — use this if you're calling it from FastAPI."""
    pdf_bytes = pdf_path.read_bytes()
    start = perf_counter()
    result = await extraction_agent.run(
        [
            "Extract full bibliographic metadata from this document.",
            BinaryContent(data=pdf_bytes, media_type="application/pdf"),
        ]
    )
    elapsed = perf_counter() - start
    result.output.original_filename = pdf_path.name
    result.output.file_size_bytes = len(pdf_bytes)
    result.output.run_time_seconds = elapsed
    result.output.input_tokens = result.usage.input_tokens
    result.output.output_tokens = result.usage.output_tokens
    result.output.total_tokens = result.usage.total_tokens
    if rename:
        _rename_pdf(pdf_path, result.output.suggested_filename)
    return result.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metadata from a PDF")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "--rename", action="store_true", help="Rename the PDF to its suggested filename"
    )
    args = parser.parse_args()
    meta = extract_metadata(args.pdf_path, rename=args.rename)
    print(meta.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
