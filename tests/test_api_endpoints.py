import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["lancedb_ready"] is True
    assert "llm_configured" in data


def test_load_sample_and_ast_endpoints():
    # 1. Load sample document (uses unified pipeline without hardcoded AST overrides)
    res = client.post("/load-sample")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "total_ast_nodes" in data
    assert "indexing_duration_ms" in data
    assert data["total_ast_nodes"] >= 4
    assert data["cross_modal_links"] >= 1

    # 2. Query AST endpoint
    ast_res = client.get("/api/ast")
    assert ast_res.status_code == 200
    ast_data = ast_res.json()
    assert "total_nodes" in ast_data
    assert "nodes" in ast_data
    assert "cross_modal_links" in ast_data
    assert "cross_modal_links_count" in ast_data
    assert ast_data["cross_modal_links_count"] == len(ast_data["cross_modal_links"])

    # 3. Query Discrepancies endpoint (verifies Table 1 98.20% vs 99.9% SLA breach is automatically detected)
    disc_res = client.get("/api/discrepancies")
    assert disc_res.status_code == 200
    disc_data = disc_res.json()
    assert "consistency_score" in disc_data
    assert disc_data["verdict"] == "FLAGGED"
    assert disc_data["contradictions_count"] >= 1
    assert len(disc_data["findings"]) >= 1


def test_upload_custom_document_and_guardrails():
    # Reject unsupported file extension
    bad_ext = client.post(
        "/upload",
        files={"file": ("malware.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    )
    assert bad_ext.status_code == 400

    # Reject empty file
    empty_res = client.post(
        "/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    )
    assert empty_res.status_code == 400

    # Upload a custom contract with a table discrepancy
    custom_doc = """1. SERVICE AVAILABILITY
1.1 Uptime Commitment: Vendor guarantees 99.9% monthly uptime across all regions.
Table 1: Regional Uptime Log:
| Region | Target | Actual | Status |
| US-East | 99.9% | 97.50% | Breach |
"""
    up_res = client.post(
        "/upload",
        files={"file": ("custom_sla.txt", io.BytesIO(custom_doc.encode("utf-8")), "text/plain")}
    )
    assert up_res.status_code == 200
    up_data = up_res.json()
    assert up_data["ok"] is True
    assert up_data["cross_modal_links"] >= 1

    disc_res = client.get("/api/discrepancies")
    assert disc_res.status_code == 200
    assert disc_res.json()["verdict"] == "FLAGGED"


def test_vault_endpoint():
    res = client.get("/vault")
    assert res.status_code == 200
    data = res.json()
    assert "vault" in data
    assert "active_doc_id" in data


def test_compliance_audit_api_endpoint():
    client.post("/load-sample")
    res = client.get("/compliance-audit")
    assert res.status_code == 200
    data = res.json()
    assert "overall_score" in data
    assert "checklist" in data
    assert len(data["checklist"]) == 7
    assert data["remediation_playbook_available"] is True

    # Passing controls must include both 'clause_excerpt' and 'excerpt' for UI compatibility
    passes = [c for c in data["checklist"] if c["status"] == "PASS"]
    assert len(passes) >= 4
    assert passes[0]["excerpt"] != ""
    assert passes[0]["clause_excerpt"] == passes[0]["excerpt"]

    # Gaps must include redline diffs
    gaps = [c for c in data["checklist"] if c["status"] == "GAP"]
    assert len(gaps) > 0
    assert gaps[0]["redline_diff"] is not None
    assert "+ " in gaps[0]["redline_diff"]


def test_export_report_markdown_and_json():
    payload = {
        "query": "What is the penalty credit?",
        "answer": "A 15% billing credit applies [1].",
        "sources": [{"citation_id": 1, "filename": "sample.txt", "page": 1, "dense_score": 0.85, "parent_text": "15% billing credit"}],
        "classification": {"label": "Legal Agreement / SLA", "type": "LEGAL_CONTRACT"},
        "evaluation": {"badge": "🛡️ GROUNDED", "confidence": 85.0},
        "baseline": "Plain baseline text.",
        "format": "markdown"
    }
    md_res = client.post("/export-report", json=payload)
    assert md_res.status_code == 200
    assert "AuditGuard RAG" in md_res.text
    assert "15% billing credit" in md_res.text

    payload["format"] = "json"
    json_res = client.post("/export-report", json=payload)
    assert json_res.status_code == 200
    parsed = json_res.json()
    assert parsed["query"] == "What is the penalty credit?"
    assert len(parsed["sources"]) == 1


def test_model_toggle_updates_health():
    initial_model = client.get("/health").json()["model"]
    toggle_res = client.post("/api/model/toggle")
    assert toggle_res.status_code == 200
    new_model = toggle_res.json()["model"]
    assert new_model != initial_model
    assert client.get("/health").json()["model"] == new_model
    # Toggle back
    client.post("/api/model/toggle")


def test_ask_and_query_endpoints_benchmark_fields():
    client.post("/load-sample")
    res = client.post("/ask", json={"query": "What is the penalty credit if availability drops below 99.5%?", "include_baseline": False})
    assert res.status_code == 200
    data = res.json()
    assert "retrieval_ms" in data
    assert "total_ms" in data
    assert data["retrieval_ms"] >= 0
    assert data["total_ms"] >= data["retrieval_ms"]
    assert "status" in data
    assert "evaluation" in data
    assert "token_telemetry" in data
