import pytest
from fastapi.testclient import TestClient
from app.main import app, DOCUMENT_VAULT, SESSION_REGISTRY

client = TestClient(app)

def test_session_isolation_and_selection():
    """
    Issue 5 Test:
    Verifies that session document selection is isolated per-session,
    preventing concurrent users from overriding each other's active document.
    """
    # 1. Load sample document to populate vault
    res = client.post("/load-sample")
    assert res.status_code == 200
    assert "sample_enterprise_sop" in DOCUMENT_VAULT

    # 2. Select document for session 1
    res1 = client.post("/session/select", json={
        "session_id": "user_session_1",
        "doc_id": "sample_enterprise_sop"
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["session_id"] == "user_session_1"
    assert data1["active_doc_id"] == "sample_enterprise_sop"

    # 3. Check session status
    status_res = client.get("/session/user_session_1")
    assert status_res.status_code == 200
    assert status_res.json()["active_doc_id"] == "sample_enterprise_sop"
    assert status_res.json()["bound"] is True

    # 4. Query with session_id
    query_res = client.post("/ask", json={
        "query": "What is the penalty credit?",
        "session_id": "user_session_1",
        "include_baseline": False
    })
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["doc_id"] == "sample_enterprise_sop"
    assert "citation_verification" in q_data

def test_query_request_per_request_doc_override():
    """
    Issue 5 Test:
    Verifies that a query can specify doc_id directly, providing complete isolation.
    """
    res = client.post("/load-sample")
    assert res.status_code == 200
    query_res = client.post("/ask", json={
        "query": "What is the uptime guarantee?",
        "doc_id": "sample_enterprise_sop",
        "include_baseline": False
    })
    assert query_res.status_code == 200
    assert query_res.json()["doc_id"] == "sample_enterprise_sop"
