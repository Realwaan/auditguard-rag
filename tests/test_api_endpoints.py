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
    assert "lancedb_ready" in data
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

    # 3. Query Discrepancies endpoint
    disc_res = client.get("/api/discrepancies")
    assert disc_res.status_code == 200
    disc_data = disc_res.json()
    assert "consistency_score" in disc_data
    assert "verdict" in disc_data
    assert "findings" in disc_data

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
    assert disc_res.json()["verdict"] in ("FLAGGED", "UNVERIFIED", "CONSISTENT")

def test_model_toggle_updates_health():
    initial_model = client.get("/health").json()["model"]
    toggle_res = client.post("/api/model/toggle")
    assert toggle_res.status_code == 200
    new_model = toggle_res.json()["model"]
    assert new_model != initial_model
    assert client.get("/health").json()["model"] == new_model
    # Toggle back
    client.post("/api/model/toggle")

def test_vault_endpoint():
    res = client.get("/vault")
    assert res.status_code == 200
    data = res.json()
    assert "vault" in data
    assert "active_doc_id" in data

def test_compliance_audit_api_endpoint():
    # Verify the compliance audit endpoint returns 7 controls and redline diffs
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
    assert len(passes) >= 1
    assert passes[0]["excerpt"] != ""
    assert passes[0]["clause_excerpt"] == passes[0]["excerpt"]

    # Gaps must include redline diffs
    gaps = [c for c in data["checklist"] if c["status"] == "GAP"]
    assert len(gaps) > 0
    assert gaps[0]["redline_diff"] is not None
    assert "+ " in gaps[0]["redline_diff"]

def test_ask_and_query_endpoints_benchmark_fields():
    client.post("/load-sample")
    res = client.post("/ask", json={"query": "What is the liability cap?", "include_baseline": False})
    assert res.status_code == 200
    data = res.json()
    assert "retrieval_ms" in data
    assert "total_ms" in data
    assert data["retrieval_ms"] >= 0
    assert data["total_ms"] >= data["retrieval_ms"]
    assert "status" in data
    assert "evaluation" in data
    assert "token_telemetry" in data

def test_export_report_markdown_and_json():
    # Test Markdown export
    md_res = client.post("/export-report", json={
        "query": "What is the uptime SLA?",
        "answer": "Guaranteed 99.9% uptime [1].",
        "sources": [{"citation_id": 1, "filename": "sla.txt", "page": 1, "parent_text": "99.9% uptime"}],
        "classification": {"label": "Legal SLA"},
        "evaluation": {"badge": "GROUNDED", "confidence": 98.5},
        "baseline": "Ungrounded output",
        "format": "markdown"
    })
    assert md_res.status_code == 200
    assert "attachment; filename=" in md_res.headers.get("Content-Disposition", "")
    assert "# 🛡️ AuditGuard" in md_res.text

    # Test JSON export
    json_res = client.post("/export-report", json={
        "query": "What is the uptime SLA?",
        "answer": "Guaranteed 99.9% uptime [1].",
        "format": "json"
    })
    assert json_res.status_code == 200
    assert "attachment; filename=" in json_res.headers.get("Content-Disposition", "")
    assert json_res.json()["title"] == "AuditGuard Grounded Audit Trail Report"
