import asyncio
import importlib.util
import json
import sys
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
    pdf.write_bytes(b"%PDF-1.4 test")
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
    assert prompt[1].data == pdf.read_bytes()
    assert prompt[1].media_type == "application/pdf"


def test_extract_metadata_async_sends_pdf_and_returns_output(
    extraction_module, tmp_path, monkeypatch
):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 async test")
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
    assert prompt[1].data == pdf.read_bytes()
    assert prompt[1].media_type == "application/pdf"


@pytest.mark.parametrize("async_mode", [False, True])
def test_rename_pdf_keeps_original_filename(extraction_module, tmp_path, async_mode):
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
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
    assert (tmp_path / "A_Book.pdf").read_bytes() == b"%PDF-1.4 test"


def test_rename_pdf_refuses_existing_destination(extraction_module, tmp_path):
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"original")
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

    assert pdf.read_bytes() == b"original"
    assert destination.read_bytes() == b"existing"


def test_rename_pdf_rejects_path_components(extraction_module, tmp_path):
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"original")
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
    pdf.write_bytes(b"%PDF-1.4 test")
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
    pdf.write_bytes(b"%PDF-1.4 test")
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
