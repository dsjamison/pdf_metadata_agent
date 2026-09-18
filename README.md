# PDF Metadata Extraction Agent

A small [PydanticAI](https://ai.pydantic.dev/) agent that reads a PDF and
returns validated, typed bibliographic metadata — title, authors, ISBNs,
publisher, edition, and a standardized suggested filename.

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
| `isbns`               | `list[str]`   | Digits only, dashes stripped                 |
| `publisher`           | `str \| None`  |                                              |
| `publication_date`     | `str \| None`  | e.g. `"2023"` or `"2023-05"`                 |
| `edition`             | `str \| None`  |                                              |
| `language`             | `str \| None`  |                                              |
| `page_count`           | `int \| None`  |                                              |
| `confidence`           | `float`       | Agent's own confidence, `0.0`–`1.0`          |
| `suggested_filename`    | `str`         | e.g. `Smith_-_Deep_Learning_(2023)`          |

## Requirements

- Python 3.10+ (for `str | None` union syntax)
- An API key for whichever model provider you use (defaults to Anthropic)

## Setup

Using `uv`:

```bash
uv add pydantic-ai
export ANTHROPIC_API_KEY=your-key-here   # or OPENAI_API_KEY / GEMINI_API_KEY etc.
```

Using `pip`:

```bash
pip install pydantic-ai
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

### Command line

```bash
python pdf_metadata_agent.py /path/to/book.pdf
```

Prints the extracted metadata as formatted JSON:

```json
{
  "title": "Deep Learning",
  "subtitle": null,
  "authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"],
  "isbns": ["9780262035613"],
  "publisher": "MIT Press",
  "publication_date": "2016",
  "edition": null,
  "language": "en",
  "page_count": 800,
  "confidence": 0.95,
  "suggested_filename": "Goodfellow_et_al_-_Deep_Learning_(2016)"
}
```

### As a library

```python
from pathlib import Path
from pdf_metadata_agent import extract_metadata

meta = extract_metadata(Path("book.pdf"))
print(meta.suggested_filename)
```

### Async (e.g. inside FastAPI)

```python
from pdf_metadata_agent import extract_metadata_async

@app.post("/extract")
async def extract(pdf_path: str):
    meta = await extract_metadata_async(Path(pdf_path))
    return meta.model_dump()
```

## Swapping models

Change the `model=` string in `extraction_agent`:

```python
extraction_agent = Agent(
    model="openai:gpt-5.2",       # or "google-gla:gemini-2.5-flash", etc.
    output_type=BookMetadata,
    ...
)
```

No other code changes are needed — PydanticAI normalizes the multimodal
input and structured-output handling across providers.

## Known limitations

- **Large PDFs**: very large files sent inline can hit provider token/size
  limits. For long books, consider sending only the first few pages, or
  use `DocumentUrl` instead of `BinaryContent` if the file is hosted
  somewhere reachable by the model provider.
- **Confidence is self-reported**: the `confidence` field is the model's
  own estimate, not a calibrated probability — treat it as a rough
  routing signal (e.g., "flag anything under 0.7 for manual review"),
  not ground truth.
