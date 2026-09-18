import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import dotenv
import pydantic_ai
import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent


@pytest.fixture
def extraction_module(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("PDF_METADATA_MODEL", raising=False)
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


def test_extract_metadata_sends_pdf_and_returns_output(extraction_module, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    expected = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book", file_size_bytes=999
    )
    extraction_module.extraction_agent.run_sync = Mock(
        return_value=SimpleNamespace(output=expected)
    )

    actual = extraction_module.extract_metadata(pdf)

    assert actual is expected
    assert actual.file_size_bytes == len(pdf.read_bytes())
    prompt = extraction_module.extraction_agent.run_sync.call_args.args[0]
    assert prompt[0] == "Extract full bibliographic metadata from this document."
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].data == pdf.read_bytes()
    assert prompt[1].media_type == "application/pdf"


def test_extract_metadata_async_sends_pdf_and_returns_output(
    extraction_module, tmp_path
):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 async test")
    expected = extraction_module.BookMetadata(
        title="A Book", confidence=0.8, suggested_filename="A_Book"
    )
    extraction_module.extraction_agent.run = AsyncMock(
        return_value=SimpleNamespace(output=expected)
    )

    actual = asyncio.run(extraction_module.extract_metadata_async(pdf))

    assert actual is expected
    assert actual.file_size_bytes == len(pdf.read_bytes())
    prompt = extraction_module.extraction_agent.run.call_args.args[0]
    assert isinstance(prompt[1], BinaryContent)
    assert prompt[1].data == pdf.read_bytes()
    assert prompt[1].media_type == "application/pdf"
