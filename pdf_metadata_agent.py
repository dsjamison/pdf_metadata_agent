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

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

load_dotenv(Path(__file__).with_name(".env"))


class BookMetadata(BaseModel):
    title: str
    subtitle: str | None = None
    authors: list[str] = Field(default_factory=list)
    isbns: list[str] = Field(
        default_factory=list, description="All ISBN-10/ISBN-13 found, digits only"
    )
    publisher: str | None = None
    publication_date: str | None = Field(
        default=None, description="Best available date, e.g. '2023' or '2023-05'"
    )
    edition: str | None = None
    language: str | None = None
    page_count: int | None = None
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
    model=os.getenv("PDF_METADATA_MODEL", "anthropic:claude-sonnet-4-6"),
    output_type=BookMetadata,
    system_prompt=(
        "You extract bibliographic metadata from book/document PDFs. "
        "Look at the title page, copyright page, and cover for ISBNs, "
        "author names, publisher, and edition. If a field truly isn't "
        "present anywhere in the document, leave it null rather than "
        "guessing. Normalize ISBNs to digits only (strip dashes/spaces).\n\n"
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


def extract_metadata(pdf_path: Path) -> BookMetadata:
    """Run the agent once on a single PDF and return validated metadata."""
    pdf_bytes = pdf_path.read_bytes()
    result = extraction_agent.run_sync(
        [
            "Extract full bibliographic metadata from this document.",
            BinaryContent(data=pdf_bytes, media_type="application/pdf"),
        ]
    )
    return result.output  # already validated as BookMetadata


async def extract_metadata_async(pdf_path: Path) -> BookMetadata:
    """Async version — use this if you're calling it from FastAPI."""
    pdf_bytes = pdf_path.read_bytes()
    result = await extraction_agent.run(
        [
            "Extract full bibliographic metadata from this document.",
            BinaryContent(data=pdf_bytes, media_type="application/pdf"),
        ]
    )
    return result.output


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_metadata_agent.py <path-to-pdf>")
        raise SystemExit(1)

    meta = extract_metadata(Path(sys.argv[1]))
    print(meta.model_dump_json(indent=2))
