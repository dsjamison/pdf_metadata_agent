# Repository Guidelines

## Project Structure & Module Organization

`pdf_metadata_agent.py` contains the PDF extraction logic, `BookMetadata` schema, and direct command-line entry point. `src/pdf_metadata_agent/__init__.py` is a separate package scaffold; its `main()` currently prints a greeting. `tests/test_pdf_metadata_agent.py` covers the schema and extraction calls. `pyproject.toml` defines dependencies and the console script. `README.md` documents extraction behavior and usage.

## Build, Test, and Development Commands

- `uv sync` creates the project environment and installs runtime and development dependencies.
- `uv run pytest -q` runs the test suite without calling a model provider.
- `python pdf_metadata_agent.py /path/to/book.pdf` runs extraction and prints JSON. Set the selected provider's API key first, such as `ANTHROPIC_API_KEY` for the current model.
- `uv build` builds the package declared in `pyproject.toml`. The installed `pdf-metadata-agent` command currently runs the scaffold in `src/`, not the extraction script.

The project declares Python 3.13 or newer in `pyproject.toml`. No lint or format command is configured.

## Coding Style & Naming Conventions

Use four-space indentation, standard Python naming (`snake_case` for functions and modules, `PascalCase` for models), and type annotations for public functions. Keep metadata fields in `BookMetadata` and give optional fields explicit defaults. Match the existing short docstring style. Avoid committing API keys or PDFs that contain private material.

## Testing Guidelines

Use `pytest` for tests under `tests/`, named `test_*.py`. Cover schema validation and both sync and async extraction paths when changing them. Stub agent responses so routine tests need no API key or provider calls. Run `uv run pytest -q` before committing. There is no coverage threshold; check a real PDF manually when changing provider integration.

## Commit & Pull Request Guidelines

The history currently contains only `Initial Commit`, so no detailed convention is established. Use short imperative subjects, such as `Add metadata validation tests`, and keep commits focused. In pull requests, describe the behavior changed, note the tests run, and mention any required provider configuration. Include a sample JSON result when extraction output changes, with personal document details removed.
