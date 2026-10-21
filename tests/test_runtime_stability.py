import pytest
import io
from fastapi import HTTPException
from fastapi.testclient import TestClient
import pypdfium2 as pdfium

from app.main import app, get_doc_context, DOCUMENT_VAULT, SESSION_REGISTRY
from app.ingest import extract_pages_from_pdf
from app.ast_builder import extract_halo_crops

client = TestClient(app)

def test_ask_stream_success_and_refusal():
    # 1. First ensure sample document is loaded
    load_res = client.post("/load-sample")
    assert load_res.status_code == 200

    # 2. Test in-bounds stream query
    stream_res = client.get("/ask-stream", params={"query": "What is the guaranteed service uptime percentage?"})
    assert stream_res.status_code == 200
    content = stream_res.text
    assert len(content) > 0
    assert "99.9%" in content or "uptime" in content.lower()

    # 3. Test out-of-bounds stream query (triggers refusal protocol)
    unverifiable_res = client.get("/ask-stream", params={"query": "What is the secret recipe for dark chocolate soufflé?"})
    assert unverifiable_res.status_code == 200
    unv_text = unverifiable_res.text
    assert "⚠️ UNVERIFIABLE:" in unv_text or "Refusal protocol activated" in unv_text

def test_scanned_pdf_graceful_handling():
    # Create an in-memory blank/scanned PDF without text streams using pypdfium2
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=200, height=200)
    pdf.new_page(width=200, height=200)
    
    bio = io.BytesIO()
    pdf.save(bio)
    pdf_bytes = bio.getvalue()
    pdf.close()

    # Extract pages from the empty/scanned PDF
    pages = extract_pages_from_pdf(pdf_bytes, "scanned_doc.pdf")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "Visual / Scanned Content" in pages[0]["text"]
    assert pages[1]["page"] == 2


def test_mixed_pdf_preserves_image_only_pages(monkeypatch):
    """Verifies that in mixed PDFs (some pages text, some pages scanned/image), textless pages are preserved."""
    from unittest.mock import patch, MagicMock

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=200, height=200)
    pdf.new_page(width=200, height=200)
    bio = io.BytesIO()
    pdf.save(bio)
    pdf_bytes = bio.getvalue()
    pdf.close()

    class MockPage:
        def __init__(self, text):
            self._text = text
        def extract_text(self):
            return self._text

    mock_reader = MagicMock()
    mock_reader.pages = [MockPage("Page 1 has SLA terms."), MockPage("")]

    with patch("app.ingest.PdfReader", return_value=mock_reader):
        pages = extract_pages_from_pdf(pdf_bytes, "mixed_sop.pdf")
        assert len(pages) == 2
        assert pages[0]["page"] == 1
        assert "Page 1 has SLA terms." in pages[0]["text"]
        assert pages[1]["page"] == 2
        assert "Visual / Scanned Content" in pages[1]["text"]
        assert pages[1].get("is_scanned") is True

    # Also verify that when OCR returns text, it is integrated
    with patch("app.ingest.PdfReader", return_value=mock_reader), \
         patch("app.ingest.ocr_extract_page_image", return_value="RECOVERED OCR CHART METRICS"):
        pages_ocr = extract_pages_from_pdf(pdf_bytes, "mixed_sop.pdf")
        assert len(pages_ocr) == 2
        assert "RECOVERED OCR CHART METRICS" in pages_ocr[1]["text"]
        assert pages_ocr[1].get("is_ocr") is True


def test_extract_halo_crops_native_cleanup():
    # Create a synthetic PDF and run halo cropping to verify native handles close cleanly
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width=300, height=400)
    bio = io.BytesIO()
    pdf.save(bio)
    pdf_bytes = bio.getvalue()
    pdf.close()

    nodes = extract_halo_crops("test_cleanup_doc", pdf_bytes)
    assert isinstance(nodes, list)

def test_session_isolation_in_audit_and_discrepancy_endpoints():
    client.post("/load-sample")

    # Bind a mock session to sample_enterprise_sop
    SESSION_REGISTRY["test_session_isolated"] = {
        "active_doc_id": "sample_enterprise_sop"
    }

    # Test /compliance-audit with session_id parameter
    audit_res = client.get("/compliance-audit", params={"session_id": "test_session_isolated"})
    assert audit_res.status_code == 200
    data = audit_res.json()
    assert data["doc_id"] == "sample_enterprise_sop"
    assert data["total_checks"] == 7

    # Test /api/discrepancies with session_id parameter
    disc_res = client.get("/api/discrepancies", params={"session_id": "test_session_isolated"})
    assert disc_res.status_code == 200
    disc_data = disc_res.json()
    assert disc_data["doc_id"] == "sample_enterprise_sop"
    assert "consistency_score" in disc_data

    # Test /api/ast with session_id parameter
    ast_res = client.get("/api/ast", params={"session_id": "test_session_isolated"})
    assert ast_res.status_code == 200
    ast_data = ast_res.json()
    assert ast_data["doc_id"] == "sample_enterprise_sop"

def test_get_doc_context_resolution_priority():
    client.post("/load-sample")
    target_id, vault_entry, retriever, classification, ast = get_doc_context(
        doc_id="sample_enterprise_sop"
    )
    assert target_id == "sample_enterprise_sop"
    assert retriever is not None
    assert ast is not None

    # Strict isolation: invalid explicit doc_id raises 404 instead of silent fallback
    with pytest.raises(HTTPException) as exc_info:
        get_doc_context(doc_id="nonexistent_ghost_doc")
    assert exc_info.value.status_code == 404

    # Strict isolation: registered session pointing to missing document raises 404
    SESSION_REGISTRY["corrupt_session"] = {"active_doc_id": "deleted_document"}
    with pytest.raises(HTTPException) as exc_info:
        get_doc_context(session_id="corrupt_session")
    assert exc_info.value.status_code == 404

    # API route returns 404 when querying an invalid explicit doc_id
    res = client.post("/ask", json={"query": "test query", "doc_id": "nonexistent_ghost_doc"})
    assert res.status_code == 404

