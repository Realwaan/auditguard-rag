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
