# Repository Guidelines

## Project Structure & Module Organization

`pdf_metadata_agent.py` contains the PDF extraction logic, `BookMetadata` schema, and direct command-line entry point. `src/pdf_metadata_agent/__init__.py` is a separate package scaffold; its `main()` currently prints a greeting. `tests/test_pdf_metadata_agent.py` covers the schema and extraction calls. `pyproject.toml` defines dependencies and the console script. `README.md` documents extraction behavior and usage.

## Build, Test, and Development Commands

- `uv sync` creates the project environment and installs runtime and development dependencies.
- `uv run pytest -q` runs the test suite without calling a model provider.
- `uv run ruff check .` checks Python code; `uv run ruff format --check .` checks formatting.
- `uv run python pdf_metadata_agent.py /path/to/book.pdf` extracts metadata and prints JSON. Add `--rename` to rename the PDF to its suggested basename plus its original extension. Configure `.env` first.
- `uv build` builds the package declared in `pyproject.toml`. The installed `pdf-metadata-agent` command currently runs the scaffold in `src/`, not the extraction script.

The project declares Python 3.13 or newer in `pyproject.toml`.

## Coding Style & Naming Conventions

Use four-space indentation, standard Python naming (`snake_case` for functions and modules, `PascalCase` for models), and type annotations for public functions. Format with `uv run ruff format .` and fix lint findings with `uv run ruff check .`. Ruff targets Python 3.13 and checks common errors and import order. Keep metadata fields in `BookMetadata` and give optional fields explicit defaults.

## Configuration & Secrets

Copy `.env.example` to `.env`. Set `PDF_METADATA_MODEL` to a PydanticAI model string and fill in its provider key: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`. The script loads `.env` beside itself; exported environment variables take precedence. `.env` is Git-ignored. Never commit keys or private PDFs.

## Testing Guidelines

Use `pytest` for tests under `tests/`, named `test_*.py`. Current tests cover schema validation, `.env` model selection and precedence, and both sync and async extraction paths. Mock agent calls so routine tests need no API key or provider calls. Run pytest and both Ruff checks before committing. There is no coverage threshold; check a real PDF manually when changing provider integration.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Add ruff for linting and formatting; update documentation`. Keep commits focused. In pull requests, describe the behavior changed, note the tests and Ruff checks run, and mention any required provider configuration. Include a redacted JSON result when extraction output changes.

## Code Map

- `src` - application source
