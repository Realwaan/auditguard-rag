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

def test_load_sample_and_ast_endpoints():
    # 1. Load sample document
    res = client.post("/load-sample")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "total_ast_nodes" in data
    assert "indexing_duration_ms" in data
    assert data["total_ast_nodes"] >= 4

    # 2. Query AST endpoint
    ast_res = client.get("/api/ast")
    assert ast_res.status_code == 200
    ast_data = ast_res.json()
    assert "total_nodes" in ast_data
    assert "nodes" in ast_data
    assert "cross_modal_links" in ast_data

    # 3. Query Discrepancies endpoint
    disc_res = client.get("/api/discrepancies")
    assert disc_res.status_code == 200
    disc_data = disc_res.json()
    assert "consistency_score" in disc_data
    assert "verdict" in disc_data
    assert "findings" in disc_data

def test_vault_endpoint():
    res = client.get("/vault")
    assert res.status_code == 200
    data = res.json()
    assert "vault" in data
    assert "active_doc_id" in data

def test_compliance_audit_api_endpoint():
    # Verify the compliance audit endpoint returns 7 controls and redline diffs
    res = client.get("/compliance-audit")
    assert res.status_code == 200
    data = res.json()
    assert "overall_score" in data
    assert "checklist" in data
    assert len(data["checklist"]) == 7
    assert data["remediation_playbook_available"] is True
    # At least one control should have a generated redline diff
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
