import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.generator import REFUSAL_TEXT, stream_grounded_answer
from tests.test_answer_regressions import evidence


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "DOCUMENT_VAULT", {})
    monkeypatch.setattr(main, "DOCUMENT_ASTS", {})
    monkeypatch.setattr(main, "SESSION_REGISTRY", {})
    monkeypatch.setattr(main, "ACTIVE_DOC_ID", "")
    monkeypatch.setattr(main, "ACTIVE_CHUNKS", [])
    monkeypatch.setattr(main, "ACTIVE_RETRIEVER", None)
    monkeypatch.setattr(main, "ACTIVE_CLASSIFICATION", {"label": "No document", "prompt_addon": ""})
    return TestClient(main.app)


@pytest.mark.parametrize("query", ["hi", "Hello!", " good morning "])
def test_greeting_does_not_retrieve_or_call_a_provider(client, monkeypatch, query):
    def unexpected(*args, **kwargs):
        raise AssertionError("Greeting entered RAG pipeline")
    monkeypatch.setattr(main, "execute_rag_retrieval_and_eval", unexpected)
    monkeypatch.setattr(main, "generate_grounded_answer", unexpected)
    monkeypatch.setattr(main, "call_plain_gemma", unexpected)
    result = client.post("/ask", json={"query": query}).json()
    assert result["generation_mode"] == "greeting"
    assert result["sources"] == []
    assert result["evaluation"]["claim_evidence_verified"] is False
    assert result["tokens"] == 0
    assert "Hello!" in client.get("/ask-stream", params={"query": query}).text
    assert main.greeting_response("Hi, what is the SLA?", "", {}) is None
    assert main.greeting_response("hi2", "", {}) is None


@pytest.mark.parametrize("ext", ["docx", "pptx", "pdf", "doc", "ppt"])
def test_malformed_or_unsupported_upload_returns_client_error(client, ext):
    result = client.post("/upload", files={"file": ("broken." + ext, io.BytesIO(b"not a document"))})
    assert result.status_code == 400
    assert main.DOCUMENT_VAULT == {}


def test_whitespace_and_oversize_upload_do_not_replace_document(client, monkeypatch):
    assert client.post("/upload", files={"file": ("empty.txt", b" \n\t ")}).status_code == 400
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE_BYTES", 10)
    assert client.post("/upload", files={"file": ("big.txt", b"x" * 11)}).status_code == 413
    assert main.ACTIVE_DOC_ID == ""


def test_failed_indexing_preserves_active_document(client, monkeypatch):
    # 1. Load a valid document first and record state
    upload1 = client.post("/upload", files={"file": ("initial.txt", b"Initial document content. Support response time is 4 hours.")})
    assert upload1.status_code == 200
    initial_doc_id = main.ACTIVE_DOC_ID
    assert initial_doc_id != ""
    initial_vault = dict(main.DOCUMENT_VAULT)
    initial_asts = dict(main.DOCUMENT_ASTS)
    initial_classification = main.ACTIVE_CLASSIFICATION
    initial_retriever = main.ACTIVE_RETRIEVER

    # Verify initial document is queryable
    q1 = client.post("/ask", json={"query": "What is support response time?", "include_baseline": False})
    assert q1.status_code == 200

    # 2. Simulate failed upload on a subsequent document
    original_embed = main.embed_texts
    def fail(_):
        raise RuntimeError("Embedding failure")
    monkeypatch.setattr(main, "embed_texts", fail)
    result = client.post("/upload", files={"file": ("corrupt.txt", b"New content that fails during indexing.")})
    assert result.status_code == 503

    # 3. Assert active document and state remain unchanged
    assert main.ACTIVE_DOC_ID == initial_doc_id
    assert main.DOCUMENT_VAULT == initial_vault
    assert main.DOCUMENT_ASTS == initial_asts
    assert main.ACTIVE_CLASSIFICATION is initial_classification
    assert main.ACTIVE_RETRIEVER is initial_retriever

    # 4. Verify initial document remains queryable after the failed upload attempt
    monkeypatch.setattr(main, "embed_texts", original_embed)
    q2 = client.post("/ask", json={"query": "What is support response time?", "include_baseline": False})
    assert q2.status_code == 200


@pytest.mark.parametrize("answer,expected", [
    (REFUSAL_TEXT, "REFUSE"),
    ("Support response time is 8 hours [1].", "CAUTION_UNSUPPORTED_CITATION"),
    ("Support response time is 4 hours [1].", "SUCCESS"),
])
def test_api_applies_final_answer_checks(client, monkeypatch, answer, expected):
    chunks = [evidence("Support response time is 4 hours.")]
    monkeypatch.setattr(main, "ACTIVE_RETRIEVER", object())
    monkeypatch.setattr(main, "execute_rag_retrieval_and_eval", lambda *a, **k: (
        chunks, {"status": "SYNTHESIZE", "confidence": 80, "grounding_verdict": "SUPPORTED",
                 "nli_verdict": "ENTAILED", "claim_evidence_verified": True}, 1.0))
    seen = []
    def generate(q, c, persona, model):
        seen.append(model)
        return {"answer": answer, "tokens": 40}
    monkeypatch.setattr(main, "generate_grounded_answer", generate)
    result = client.post("/ask", json={"query": "What is support response time?", "include_baseline": False}).json()
    assert result["status"] == expected
    assert seen == [None]  # Each provider selects its own configured model.
    assert result["evaluation"]["claim_evidence_verified"] is False
    assert result["token_telemetry"]["estimated"] is False
    if expected != "SUCCESS":
        assert result["evaluation"]["grounding_verdict"] != "SUPPORTED"
        assert "ENTAILED" not in result["evaluation"]["nli_verdict"]


def test_alma_model_and_explicit_override(monkeypatch):
    import app.generator as generator
    import app.llm as llm
    monkeypatch.setattr(generator, "USE_GOOGLE_GENAI", False)
    monkeypatch.setattr(generator, "LLM_MODEL", "venue-model-id")
    monkeypatch.setattr(llm, "USE_GOOGLE_GENAI", False)
    monkeypatch.setattr(llm, "LLM_MODEL", "venue-model-id")
    seen = []
    def create(**kwargs):
        seen.append(kwargs["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Uptime is 99.9% [1]."))], usage=None)
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(generator, "get_openai_client", lambda: provider)
    monkeypatch.setattr(llm, "get_openai_client", lambda: provider)
    generator.generate_grounded_answer("What is uptime?", [evidence("Uptime is 99.9%.")])
    llm.call_plain_gemma("What is uptime?", "selected-model")
    assert seen == ["venue-model-id", "selected-model"]
    assert llm.get_default_model() == "venue-model-id"

    # Verify streaming also defaults to provider-aware default model
    seen_stream = []
    def create_stream(**kwargs):
        seen_stream.append(kwargs["model"])
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Uptime is 99.9% [1]."))])])
    provider_stream = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_stream)))
    monkeypatch.setattr(generator, "get_openai_client", lambda: provider_stream)
    list(generator.stream_grounded_answer("What is uptime?", [evidence("Uptime is 99.9%.")]))
    list(generator.stream_grounded_answer("What is uptime?", [evidence("Uptime is 99.9%.")], model_override="custom-stream-model"))
    assert seen_stream == ["venue-model-id", "custom-stream-model"]


def test_partial_stream_never_appends_a_second_provider_answer(monkeypatch):
    import app.generator as generator
    monkeypatch.setattr(generator, "USE_GOOGLE_GENAI", True)
    def chunks(**kwargs):
        yield SimpleNamespace(text="The partial answer")
        raise RuntimeError("Stream disconnected")
    google = SimpleNamespace(models=SimpleNamespace(generate_content_stream=chunks))
    monkeypatch.setattr(generator, "get_genai_client", lambda: google)
    def unexpected():
        raise AssertionError("Must not switch providers after emitting text")
    monkeypatch.setattr(generator, "get_openai_client", unexpected)
    text = "".join(stream_grounded_answer("What is uptime?", [evidence("Uptime is 99.9%.")]))
    assert text.startswith("The partial answer")
    assert "interrupted" in text and "unverified" in text


def test_stream_skips_empty_usage_frames(monkeypatch):
    import app.generator as generator
    monkeypatch.setattr(generator, "USE_GOOGLE_GENAI", False)
    frames = [SimpleNamespace(choices=[]), SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Uptime is 99.9% [1]."))])]
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: iter(frames))))
    monkeypatch.setattr(generator, "get_openai_client", lambda: provider)
    assert "".join(stream_grounded_answer("What is uptime?", [evidence("Uptime is 99.9%.")])) == "Uptime is 99.9% [1]."


def test_lazy_preview_is_loaded_and_arbitrary_files_are_rejected(monkeypatch, tmp_path):
    import app.llm as llm
    import app.ingest as ingest
    monkeypatch.setattr(llm, "BASE_DIR", tmp_path)
    directory = tmp_path / "app" / "static" / "previews"
    directory.mkdir(parents=True)
    image = directory / "lesson_page_15.webp"
    image.write_bytes(b"test-image")
    calls = []
    def render(prefix, page):
        calls.append((prefix, page))
        return image
    monkeypatch.setattr(ingest, "get_or_render_page_image", render)
    assert llm.load_image_bytes("/api/preview/lesson/15") == (b"test-image", "image/webp")
    assert calls == [("lesson", 15)]
    assert llm.load_image_bytes("/static/../../.env") is None
    private_file = tmp_path / "private.png"
    private_file.write_bytes(b"private")
    assert llm.load_image_bytes(str(private_file)) is None


def test_stream_warns_about_incorrect_numbers(monkeypatch):
    import app.generator as generator
    monkeypatch.setattr(generator, "USE_GOOGLE_GENAI", False)
    frames = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Uptime is 95.9% [1]."))])]
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: iter(frames))))
    monkeypatch.setattr(generator, "get_openai_client", lambda: provider)
    result = "".join(stream_grounded_answer("What is uptime?", [evidence("Uptime is 99.9%.")]))
    assert "requires review" in result
