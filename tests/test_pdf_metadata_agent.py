import asyncio
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import dotenv
import pydantic_ai
import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.usage import RunUsage
from pypdf import PdfReader, PdfWriter


def _write_sample_pdf(path: Path, pages: int = 1) -> bytes:
    """Write a real multi-page PDF for tests; return the original bytes."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)
    return path.read_bytes()


@pytest.fixture
def extraction_module(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("PDF_METADATA_MODEL", raising=False)
    monkeypatch.setenv("PDF_METADATA_DB", str(tmp_path / "test.db"))
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_: False)
    script = Path(__file__).resolve().parents[1] / "pdf_metadata_agent.py"
    spec = importlib.util.spec_from_file_location(
        "pdf_metadata_extraction_script", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("environment_model", "expected_model"),
    [
        (None, "openai:gpt-5.2"),
        ("test", "test"),
    ],
)
def test_loads_model_from_dotenv_without_overriding_environment(
    monkeypatch, tmp_path, environment_model, expected_model
):
    script = tmp_path / "pdf_metadata_agent.py"
    script.write_text(
        (Path(__file__).resolve().parents[1] / "pdf_metadata_agent.py").read_text()
    )
    (tmp_path / ".env").write_text("PDF_METADATA_MODEL=openai:gpt-5.2\n")
    if environment_model is None:
        monkeypatch.delenv("PDF_METADATA_MODEL", raising=False)
    else:
        monkeypatch.setenv("PDF_METADATA_MODEL", environment_model)
    agent = Mock()
    monkeypatch.setattr(pydantic_ai, "Agent", agent)

    spec = importlib.util.spec_from_file_location("test_dotenv_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert agent.call_args.kwargs["model"] == expected_model


def test_book_metadata_defaults_and_confidence_bounds(extraction_module):
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.5, suggested_filename="A_Book"
    )

    assert metadata.authors == []
    assert metadata.isbns == []
    assert metadata.subtitle is None
    assert metadata.subjects == []
    assert metadata.keywords == []
    assert metadata.summary is None
    assert metadata.document_type is None
    assert metadata.file_size_bytes is None
    assert metadata.original_filename is None
    assert metadata.run_time_seconds is None
    assert metadata.total_tokens is None
    assert metadata.provider is None
    assert metadata.model is None

    with pytest.raises(ValidationError):
        extraction_module.BookMetadata(
            title="A Book", confidence=1.1, suggested_filename="A_Book"
        )


def test_isbn_formats_and_new_metadata_fields(extraction_module):
    metadata = extraction_module.BookMetadata.model_validate(
        {
            "title": "A Book",
            "isbns": [
                {"value": "9780262035613", "format": "cloth"},
                {"value": "9780262337373", "format": "EPUB"},
            ],
            "subjects": ["Machine learning"],
            "keywords": ["neural networks"],
            "summary": "An introduction to deep learning.",
            "document_type": "book",
            "confidence": 0.8,
            "suggested_filename": "A_Book",
        }
    )

    assert metadata.isbns[0].format == "cloth"
    assert metadata.isbns[1].value == "9780262337373"
    assert metadata.subjects == ["Machine learning"]
    assert metadata.keywords == ["neural networks"]
    assert metadata.summary == "An introduction to deep learning."
    assert metadata.document_type == "book"


def test_extract_metadata_sends_pdf_and_returns_output(
    extraction_module, tmp_path, monkeypatch
):
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    expected = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book", file_size_bytes=999
    )
    monkeypatch.setattr(extraction_module, "perf_counter", Mock(side_effect=[10, 12.5]))
    extraction_module.extraction_agent.run_sync = Mock(
        return_value=SimpleNamespace(
            output=expected, usage=RunUsage(input_tokens=30, output_tokens=12)
        )
    )

    actual = extraction_module.extract_metadata(pdf)

    assert actual is expected
    assert actual.file_size_bytes == len(pdf.read_bytes())
    assert actual.page_count == 1
    assert actual.original_filename == "book.pdf"
    assert actual.run_time_seconds == 2.5
    assert (actual.input_tokens, actual.output_tokens, actual.total_tokens) == (
        30,
        12,
        42,
    )
    prompt = extraction_module.extraction_agent.run_sync.call_args.args[0]
    assert prompt[0] == "Extract full bibliographic metadata from this document."
    assert isinstance(prompt[1], BinaryContent)
    assert len(PdfReader(BytesIO(prompt[1].data)).pages) == 1
    assert prompt[1].media_type == "application/pdf"


def test_extract_metadata_async_sends_pdf_and_returns_output(
    extraction_module, tmp_path, monkeypatch
):
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    expected = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book"
    )
    monkeypatch.setattr(
        extraction_module, "perf_counter", Mock(side_effect=[20, 21.25])
    )
    extraction_module.extraction_agent.run = AsyncMock(
        return_value=SimpleNamespace(
            output=expected, usage=RunUsage(input_tokens=8, output_tokens=4)
        )
    )

    actual = asyncio.run(extraction_module.extract_metadata_async(pdf))

    assert actual is expected
    assert actual.file_size_bytes == len(pdf.read_bytes())
    assert actual.original_filename == "book.pdf"
    assert actual.run_time_seconds == 1.25
    assert (actual.input_tokens, actual.output_tokens, actual.total_tokens) == (
        8,
        4,
        12,
    )
    prompt = extraction_module.extraction_agent.run.call_args.args[0]
    assert isinstance(prompt[1], BinaryContent)
    assert len(PdfReader(BytesIO(prompt[1].data)).pages) == 1
    assert prompt[1].media_type == "application/pdf"


@pytest.mark.parametrize("async_mode", [False, True])
def test_rename_pdf_keeps_original_filename(extraction_module, tmp_path, async_mode):
    pdf = tmp_path / "original.pdf"
    original = _write_sample_pdf(pdf)
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book"
    )
    result = SimpleNamespace(output=metadata, usage=RunUsage())

    if async_mode:
        extraction_module.extraction_agent.run = AsyncMock(return_value=result)
        actual = asyncio.run(extraction_module.extract_metadata_async(pdf, rename=True))
    else:
        extraction_module.extraction_agent.run_sync = Mock(return_value=result)
        actual = extraction_module.extract_metadata(pdf, rename=True)

    assert actual.original_filename == "original.pdf"
    assert not pdf.exists()
    assert (tmp_path / "A_Book.pdf").read_bytes() == original


def test_rename_pdf_refuses_existing_destination(extraction_module, tmp_path):
    pdf = tmp_path / "original.pdf"
    original = _write_sample_pdf(pdf)
    destination = tmp_path / "A_Book.pdf"
    destination.write_bytes(b"existing")
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book"
    )
    extraction_module.extraction_agent.run_sync = Mock(
        return_value=SimpleNamespace(output=metadata, usage=RunUsage())
    )

    with pytest.raises(FileExistsError):
        extraction_module.extract_metadata(pdf, rename=True)

    assert pdf.read_bytes() == original
    assert destination.read_bytes() == b"existing"


def test_rename_pdf_rejects_path_components(extraction_module, tmp_path):
    pdf = tmp_path / "original.pdf"
    _write_sample_pdf(pdf)
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="../elsewhere"
    )
    extraction_module.extraction_agent.run_sync = Mock(
        return_value=SimpleNamespace(output=metadata, usage=RunUsage())
    )

    with pytest.raises(ValueError, match="single nonempty basename"):
        extraction_module.extract_metadata(pdf, rename=True)

    assert pdf.exists()


def test_cli_rename_flag(extraction_module, monkeypatch, capsys):
    metadata = Mock()
    metadata.model_dump_json.return_value = "{}"
    extract = Mock(return_value=metadata)
    monkeypatch.setattr(extraction_module, "extract_metadata", extract)
    monkeypatch.setattr(sys, "argv", ["pdf_metadata_agent.py", "book.pdf", "--rename"])

    extraction_module.main()

    extract.assert_called_once_with(Path("book.pdf"), rename=True)
    assert capsys.readouterr().out == "{}\n"


def test_build_model_returns_model_string_for_default_providers(
    extraction_module, monkeypatch
):
    monkeypatch.setenv("PDF_METADATA_MODEL", "anthropic:claude-sonnet-4-6")

    assert extraction_module._build_model() == "anthropic:claude-sonnet-4-6"


def test_build_model_wires_meta_chat_model_without_forced_tools(
    extraction_module, monkeypatch
):
    monkeypatch.setenv("PDF_METADATA_MODEL", "meta:muse-spark-1.3-contributor")
    monkeypatch.setenv("META_API_KEY", "meta-test-key")

    model = extraction_module._build_model()

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "muse-spark-1.3-contributor"
    assert model.profile.get("openai_supports_tool_choice_required") is False
    assert "api.meta.ai" in model.base_url


def test_build_model_meta_requires_api_key(extraction_module, monkeypatch):
    monkeypatch.setenv("PDF_METADATA_MODEL", "meta:muse-spark-1.3-contributor")
    monkeypatch.delenv("META_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="META_API_KEY"):
        extraction_module._build_model()


@pytest.mark.parametrize(
    ("configured_model", "expected_provider", "expected_model"),
    [
        ("meta:muse-spark-1.3-contributor", "meta", "muse-spark-1.3-contributor"),
        ("anthropic:claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("localmodel", None, "localmodel"),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_extract_metadata_records_provider_and_model(
    extraction_module,
    tmp_path,
    monkeypatch,
    configured_model,
    expected_provider,
    expected_model,
    async_mode,
):
    monkeypatch.setenv("PDF_METADATA_MODEL", configured_model)
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book"
    )
    result = SimpleNamespace(output=metadata, usage=RunUsage())

    if async_mode:
        extraction_module.extraction_agent.run = AsyncMock(return_value=result)
        actual = asyncio.run(extraction_module.extract_metadata_async(pdf))
    else:
        extraction_module.extraction_agent.run_sync = Mock(return_value=result)
        actual = extraction_module.extract_metadata(pdf)

    assert actual.provider == expected_provider
    assert actual.model == expected_model


def test_save_and_list_metadata_round_trip(extraction_module):
    metadata = extraction_module.BookMetadata(
        title="A Book",
        confidence=0.8,
        suggested_filename="A_Book",
        authors=["Jane Doe"],
        isbns=[{"value": "9780262035613", "format": "cloth"}],
        subjects=["Machine learning"],
        keywords=["neural networks"],
        provider="meta",
        model="muse-spark-1.3-contributor",
    )

    row_id = extraction_module.save_metadata(metadata)

    (record,) = extraction_module.list_metadata()
    assert record["id"] == row_id
    assert record["title"] == "A Book"
    assert record["suggested_filename"] == "A_Book"
    assert record["authors"] == ["Jane Doe"]
    assert record["isbns"] == [{"value": "9780262035613", "format": "cloth"}]
    assert record["subjects"] == ["Machine learning"]
    assert record["keywords"] == ["neural networks"]
    assert record["provider"] == "meta"
    assert record["model"] == "muse-spark-1.3-contributor"
    assert record["extracted_at"]


def test_list_metadata_empty_database(extraction_module):
    assert extraction_module.list_metadata() == []


def test_extract_metadata_saves_record_to_database(
    extraction_module, tmp_path, monkeypatch
):
    monkeypatch.setenv("PDF_METADATA_MODEL", "meta:muse-spark-1.3-contributor")
    pdf = tmp_path / "book.pdf"
    _write_sample_pdf(pdf)
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.9, suggested_filename="A_Book"
    )
    extraction_module.extraction_agent.run_sync = Mock(
        return_value=SimpleNamespace(output=metadata, usage=RunUsage())
    )

    extraction_module.extract_metadata(pdf)

    (record,) = extraction_module.list_metadata()
    assert record["title"] == "A Book"
    assert record["provider"] == "meta"
    assert record["model"] == "muse-spark-1.3-contributor"


def test_cli_list_flag(extraction_module, monkeypatch, capsys):
    extraction_module.save_metadata(
        extraction_module.BookMetadata(
            title="A Book", confidence=0.5, suggested_filename="A_Book"
        )
    )
    monkeypatch.setattr(sys, "argv", ["pdf_metadata_agent.py", "--list"])

    extraction_module.main()

    (record,) = json.loads(capsys.readouterr().out)
    assert record["title"] == "A Book"


def test_cli_requires_pdf_path_without_list(extraction_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pdf_metadata_agent.py"])

    with pytest.raises(SystemExit):
        extraction_module.main()


def test_expand_paths_sorts_dedupes_and_passes_literals(extraction_module, tmp_path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    _write_sample_pdf(second)
    _write_sample_pdf(first)
    missing = tmp_path / "missing.pdf"

    result = extraction_module._expand_paths([tmp_path / "*.pdf", first, missing])

    assert result == [first, second, missing]


def test_cli_wildcard_processes_multiple_pdfs(
    extraction_module, tmp_path, monkeypatch, capsys
):
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    _write_sample_pdf(first)
    _write_sample_pdf(second)

    def fake_extract(path, rename=False):
        return extraction_module.BookMetadata(
            title=path.stem, confidence=0.9, suggested_filename=path.stem
        )

    monkeypatch.setattr(
        extraction_module, "extract_metadata", Mock(side_effect=fake_extract)
    )
    monkeypatch.setattr(sys, "argv", ["pdf_metadata_agent.py", str(tmp_path / "*.pdf")])

    extraction_module.main()

    calls = extraction_module.extract_metadata.call_args_list
    assert [call.args[0] for call in calls] == [first, second]
    records = json.loads(capsys.readouterr().out)
    assert [record["title"] for record in records] == ["a", "b"]


def test_cli_pattern_without_matches_errors(extraction_module, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["pdf_metadata_agent.py", str(tmp_path / "none-*.pdf")]
    )

    with pytest.raises(SystemExit):
        extraction_module.main()


@pytest.mark.parametrize(
    ("configured_pages", "expected_pages"),
    [(None, 10), ("3", 3), ("25", 15)],
)
@pytest.mark.parametrize("async_mode", [False, True])
def test_extract_sends_first_n_pages_and_total_count(
    extraction_module,
    tmp_path,
    monkeypatch,
    configured_pages,
    expected_pages,
    async_mode,
):
    if configured_pages is None:
        monkeypatch.delenv("PDF_METADATA_MAX_PAGES", raising=False)
    else:
        monkeypatch.setenv("PDF_METADATA_MAX_PAGES", configured_pages)
    pdf = tmp_path / "book.pdf"
    original = _write_sample_pdf(pdf, pages=15)
    metadata = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book"
    )
    result = SimpleNamespace(output=metadata, usage=RunUsage())

    if async_mode:
        extraction_module.extraction_agent.run = AsyncMock(return_value=result)
        actual = asyncio.run(extraction_module.extract_metadata_async(pdf))
    else:
        extraction_module.extraction_agent.run_sync = Mock(return_value=result)
        actual = extraction_module.extract_metadata(pdf)

    if async_mode:
        prompt = extraction_module.extraction_agent.run.call_args.args[0]
    else:
        prompt = extraction_module.extraction_agent.run_sync.call_args.args[0]
    assert len(PdfReader(BytesIO(prompt[1].data)).pages) == expected_pages
    assert actual.page_count == 15
    assert actual.file_size_bytes == len(original)


@pytest.mark.parametrize("configured_pages", ["0", "-2", "abc"])
def test_max_pages_rejects_non_positive_integers(
    extraction_module, monkeypatch, configured_pages
):
    monkeypatch.setenv("PDF_METADATA_MAX_PAGES", configured_pages)

    with pytest.raises(ValueError, match="PDF_METADATA_MAX_PAGES"):
        extraction_module._max_pages()


def test_max_pages_defaults_to_ten(extraction_module, monkeypatch):
    monkeypatch.delenv("PDF_METADATA_MAX_PAGES", raising=False)

    assert extraction_module._max_pages() == 10
