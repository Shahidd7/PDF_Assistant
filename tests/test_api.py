import io

import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pypdf import PdfWriter

import api
import rag

client = TestClient(api.app)


def blank_pdf_bytes() -> bytes:
    """A valid PDF with one empty page (no text layer, like a scan)."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def fake_docs():
    return [Document(page_content="Shadow deployment runs a new model alongside the old one.",
                     metadata={"page": 302})]


# ---------- upload validation ----------

def test_upload_rejects_non_pdf_extension():
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422
    assert "PDF" in r.json()["detail"]


def test_upload_rejects_fake_pdf_content():
    r = client.post("/api/upload", files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")})
    assert r.status_code == 422
    assert "real PDF" in r.json()["detail"]


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(api, "MAX_FILE_MB", 0)    # any content now counts as too large
    r = client.post("/api/upload", files={"file": ("big.pdf", b"%PDF-1.4 data", "application/pdf")})
    assert r.status_code == 413


def test_upload_rejects_pdf_without_text():
    r = client.post("/api/upload", files={"file": ("scan.pdf", blank_pdf_bytes(), "application/pdf")})
    assert r.status_code == 422
    assert "No text" in r.json()["detail"]


# ---------- ask validation ----------

def test_ask_unknown_document_returns_404():
    r = client.post("/api/ask", json={"doc_id": "missing", "question": "hello?"})
    assert r.status_code == 404


def test_ask_rejects_empty_question():
    r = client.post("/api/ask", json={"doc_id": "x", "question": ""})
    assert r.status_code == 422


def test_ask_rejects_too_long_question():
    r = client.post("/api/ask", json={"doc_id": "x", "question": "a" * 501})
    assert r.status_code == 422


def test_ask_rejects_whitespace_only_question(monkeypatch):
    monkeypatch.setitem(api.stores, "doc1", object())
    r = client.post("/api/ask", json={"doc_id": "doc1", "question": "     "})
    assert r.status_code == 422


# ---------- ask behaviour (retrieval and LLM mocked) ----------

def test_ask_returns_answer_pages_and_sources(monkeypatch):
    monkeypatch.setitem(api.stores, "doc1", object())
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: fake_docs())
    monkeypatch.setattr(rag, "generate", lambda q, d: {
        "answer": "It runs alongside the old model. [p. 302]",
        "latency_s": 0.1, "input_tokens": 10, "output_tokens": 5,
    })
    r = client.post("/api/ask", json={"doc_id": "doc1", "question": "What is shadow deployment?"})
    assert r.status_code == 200
    body = r.json()
    assert body["pages"] == [302]
    assert body["sources"][0]["page"] == 302
    assert "302" in body["answer"]


def _status_error(cls, status):
    resp = httpx.Response(status, request=httpx.Request("POST", "http://test"))
    return cls("boom", response=resp, body=None)


def test_rate_limit_maps_to_429(monkeypatch):
    monkeypatch.setitem(api.stores, "doc1", object())
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: fake_docs())

    def boom(q, d):
        raise _status_error(groq.RateLimitError, 429)

    monkeypatch.setattr(rag, "generate", boom)
    r = client.post("/api/ask", json={"doc_id": "doc1", "question": "anything?"})
    assert r.status_code == 429


# ---------- the refusal contract ----------

def test_refusal_sentence_is_in_the_system_prompt():
    """The eval's refusal check depends on this exact sentence reaching the model."""
    messages = rag.prompt.format_messages(context="ctx", question="q")
    system_text = messages[0].content
    assert rag.REFUSAL in system_text
    assert "ONLY" in system_text