import pytest
from app.classifier import classify_document

def test_classify_legal_contract():
    text = "SERVICE LEVEL AGREEMENT. This Agreement outlines liabilities, indemnification, and termination governing law."
    result = classify_document(text)
    assert result["type"] == "LEGAL_CONTRACT"
    assert "Legal Agreement" in result["label"]
    assert "Legal & Compliance" in result["prompt_addon"]

def test_classify_sop():
    text = "STANDARD OPERATING PROCEDURE: Incident Response Workflow. Step 1: Notify manager. Step 2: Escalation flow."
    result = classify_document(text)
    assert result["type"] == "SOP_PROCEDURAL"
    assert "Operations Director" in result["prompt_addon"]

def test_classify_financial():
    text = "Quarterly Financial Audit Report: Fiscal Year balance sheet, EBITDA, revenue, and net income."
    result = classify_document(text)
    assert result["type"] == "FINANCIAL_AUDIT"
    assert "Financial Verifier" in result["prompt_addon"]

def test_classify_academic():
    text = "University Policy Handbook: Academic integrity, student affairs, honor code, and grading appeals."
    result = classify_document(text)
    assert result["type"] == "ACADEMIC_POLICY"
    assert "Academic Policy" in result["prompt_addon"]

def test_classify_general_fallback():
    text = "Random notes about general topics."
    result = classify_document(text)
    assert result["type"] == "GENERAL_SPECIFICATION"
