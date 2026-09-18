# PDF Metadata Extraction Agent

A small [PydanticAI](https://ai.pydantic.dev/) agent that reads a PDF and
returns validated, typed bibliographic metadata — title, authors, labeled ISBNs,
subjects, summary, document type, and a standardized suggested filename. It
also reports the PDF file size, model run time, and token usage.

The PDF is sent to a multimodal LLM (Claude, GPT, Gemini, etc.) as raw
bytes — there's no manual text extraction or regex parsing. PydanticAI
validates the model's response against a `BookMetadata` schema and retries
automatically if the output doesn't conform.

## What it extracts

| Field                | Type          | Notes                                       |
|-----------------------|---------------|----------------------------------------------|
| `title`               | `str`         | Required                                     |
| `subtitle`             | `str \| None`  |                                              |
| `authors`             | `list[str]`   |                                              |
| `isbns`               | `list[ISBNEntry]` | Each has `value` (digits only) and optional `format`, such as cloth, ePDF, or EPUB |
| `publisher`           | `str \| None`  |                                              |
| `publication_date`     | `str \| None`  | e.g. `"2023"` or `"2023-05"`                 |
| `edition`             | `str \| None`  |                                              |
| `language`             | `str \| None`  |                                              |
| `page_count`           | `int \| None`  |                                              |
| `subjects`            | `list[str]`   | Document topics                              |
| `keywords`            | `list[str]`   | Search terms found in the document           |
| `summary`             | `str \| None`  | Brief description of the content             |
| `document_type`       | `str \| None`  | e.g. book, article, white paper, magazine    |
| `file_size_bytes`     | `int \| None`  | Measured from the input PDF after extraction |
| `original_filename`   | `str \| None`  | Source PDF basename before any rename        |
| `run_time_seconds`    | `float \| None` | Elapsed time for the model run                |
| `input_tokens`        | `int \| None`  | Input tokens reported by PydanticAI          |
| `output_tokens`       | `int \| None`  | Output tokens reported by PydanticAI         |
| `total_tokens`        | `int \| None`  | Sum of input and output tokens               |
| `confidence`           | `float`       | Agent's own confidence, `0.0`–`1.0`          |
| `suggested_filename`    | `str`         | e.g. `Smith_-_Deep_Learning_(2023)`          |

## Requirements

- Python 3.13+ (as declared in `pyproject.toml`)
- An API key for whichever model provider you use (defaults to Anthropic)

## Setup

Using `uv`:

```bash
uv sync
cp .env.example .env
```

Edit `.env` to set `PDF_METADATA_MODEL` and the API key for that provider.
The example lists Anthropic, OpenAI, and Gemini model strings. The program
loads `.env` next to `pdf_metadata_agent.py` at startup. Existing environment
variables take precedence. `.env` is ignored by Git; keep real keys out of
`.env.example` and commits.

Using `pip` instead:

```bash
pip install pydantic-ai python-dotenv
cp .env.example .env
```

## Usage

### Command line

```bash
uv run python pdf_metadata_agent.py /path/to/book.pdf
```

To rename the PDF after extraction, add `--rename`:

```bash
uv run python pdf_metadata_agent.py /path/to/book.pdf --rename
```

The new name is the suggested basename plus the PDF's existing extension.
Without `--rename`, the file is left in place. Renaming fails if the target
filename already exists; it does not replace that file.

Run this from the repository root. The installed `pdf-metadata-agent` console
command currently prints a scaffold greeting; use the script above for extraction.
The script prints extracted metadata as formatted JSON:

```json
{
  "title": "Deep Learning",
  "subtitle": null,
  "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
  "isbns": [{"value": "9780262035613", "format": "cloth"}],
  "publisher": "MIT Press",
  "publication_date": "2016",
  "edition": null,
  "language": "en",
  "page_count": 800,
  "subjects": ["Machine learning"],
  "keywords": ["neural networks", "deep learning"],
  "summary": "An introduction to deep learning methods and applications.",
  "document_type": "book",
  "file_size_bytes": 12345678,
  "original_filename": "book.pdf",
  "run_time_seconds": 4.27,
  "input_tokens": 2048,
  "output_tokens": 256,
  "total_tokens": 2304,
  "confidence": 0.95,
  "suggested_filename": "Goodfellow_et_al_-_Deep_Learning_(2016)"
}
```

### As a library

```python
from pathlib import Path
from pdf_metadata_agent import extract_metadata

meta = extract_metadata(Path("book.pdf"))  # pass rename=True to rename the PDF
print(meta.suggested_filename)
```

### Async (e.g. inside FastAPI)

```python
from pdf_metadata_agent import extract_metadata_async


@app.post("/extract")
async def extract(pdf_path: str):
    meta = await extract_metadata_async(Path(pdf_path), rename=False)
    return meta.model_dump()
```

## Swapping models

Set `PDF_METADATA_MODEL` in `.env` to a PydanticAI model string, for example:

```dotenv
PDF_METADATA_MODEL=openai:gpt-5.2
OPENAI_API_KEY=your-key-here
```

The default is `anthropic:claude-sonnet-4-6`. For Gemini, select
`google-gla:gemini-2.5-flash` and set `GOOGLE_API_KEY`. Other PydanticAI model
strings can be used with their provider's credentials.

For Meta's API, use the `meta:` scheme with your Meta key:

```dotenv
PDF_METADATA_MODEL=meta:muse-spark-1.3-contributor
META_API_KEY=your-key-here
```

The `meta:` scheme talks to Meta's OpenAI-compatible Chat Completions API
(`https://api.meta.ai/v1` by default, overridable with `META_BASE_URL`) and
leaves `tool_choice` on `auto`, since Meta rejects forced tool use. Do not use
the `openai:` prefix for Meta: it targets the Responses API, which Meta does
not implement.

## Tests and contributions

Install the project and its development dependencies, then run the tests:

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Tests in `tests/test_pdf_metadata_agent.py` cover metadata validation, `.env`
model selection and environment precedence, and both extraction functions.
They mock model calls, so no API key or PDF fixture is needed. Add focused
`test_*.py` tests for behavior changes. Run `uv run ruff format .` to apply
formatting, then run the checks above before committing. Use a short,
imperative commit subject such as `Add metadata validation tests`. In pull
requests, summarize the change and list the checks run; include a redacted
JSON example if the output changes.

## Known limitations

- **Large PDFs**: very large files sent inline can hit provider token/size
  limits. For long books, consider sending only the first few pages, or
  use `DocumentUrl` instead of `BinaryContent` if the file is hosted
  somewhere reachable by the model provider.
- **Confidence is self-reported**: the `confidence` field is the model's
  own estimate, not a calibrated probability — treat it as a rough
  routing signal (e.g., "flag anything under 0.7 for manual review"),
  not ground truth.
