# Repository Guidelines

## Project Structure & Module Organization

`pdf_metadata_agent.py` contains the PDF extraction logic, `BookMetadata` schema, and direct command-line entry point. `src/pdf_metadata_agent/__init__.py` is a separate package scaffold; its `main()` currently prints a greeting. `pyproject.toml` defines the package and console script. `README.md` documents extraction behavior and usage. There is no `tests/` directory or bundled PDF asset yet.

## Build, Test, and Development Commands

- `uv sync` creates the project environment from `pyproject.toml`.
- `uv add pydantic-ai` installs the extraction code's required dependency; it is not yet declared in `pyproject.toml`.
- `python pdf_metadata_agent.py /path/to/book.pdf` runs extraction and prints JSON. Set the selected provider's API key first, such as `ANTHROPIC_API_KEY` for the current model.
- `uv build` builds the package declared in `pyproject.toml`. The installed `pdf-metadata-agent` command currently runs the scaffold in `src/`, not the extraction script.

The project declares Python 3.13 or newer in `pyproject.toml`. There is no configured lint, format, or automated test command yet.

## Coding Style & Naming Conventions

Use four-space indentation, standard Python naming (`snake_case` for functions and modules, `PascalCase` for models), and type annotations for public functions. Keep metadata fields in `BookMetadata` and give optional fields explicit defaults. Match the existing short docstring style. Avoid committing API keys or PDFs that contain private material.

## Testing Guidelines

No test framework or coverage threshold is configured. For changes to extraction, add focused `pytest` tests under `tests/` named `test_*.py`; use a stubbed model response for schema and filename behavior so routine tests do not require an API key or incur provider charges. Run them with `uv run pytest` after adding `pytest` as a development dependency. Check a real PDF manually when changing provider integration.

## Commit & Pull Request Guidelines

This repository has no commits yet, so it has no established commit message convention. Use short imperative subjects, such as `Add metadata validation tests`. In pull requests, describe the behavior changed, note the commands run, and mention any required provider configuration. Include a sample JSON result when extraction output changes, with personal document details removed.
