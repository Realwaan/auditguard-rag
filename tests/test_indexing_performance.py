"""Regression coverage for search readiness, cache correctness and lazy evidence."""
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, NumberObject, DecodedStreamObject

import app.ast_builder as ast_module
import app.config as cfg
import app.embeddings as embeddings
import app.indexing as indexing
import app.ingest as ingest
import app.main as main
from app.ast_builder import ASTNodeType, DocumentAST


TEXT = b"The vendor guarantees 99.9% uptime. A 15% billing credit applies below 99.5%."
TABLE = b"""1. SERVICE AVAILABILITY
Vendor guarantees 99.9% uptime as recorded in Table 1.
Table 1: Regional Uptime Log:
| Region | Target | Actual |
| US-East | 99.9% | 97.50% |
"""


def small_pdf(page_count=2):
    """A real PDF with extractable text and a sufficiently large embedded image."""
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    picture = DecodedStreamObject()
    picture.set_data(bytes([40, 120, 200]) * 80 * 80)
    picture.update({NameObject("/Type"): NameObject("/XObject"),
                    NameObject("/Subtype"): NameObject("/Image"),
                    NameObject("/Width"): NumberObject(80), NameObject("/Height"): NumberObject(80),
                    NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                    NameObject("/BitsPerComponent"): NumberObject(8)})
    for _ in range(page_count):
        page = writer.add_blank_page(width=300, height=400)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
            NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): writer._add_object(picture)}),
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 10 Tf 20 350 Td (Figure 1: Monthly uptime is 99.9 percent.) Tj ET\nq 100 0 0 80 30 200 cm /Im1 Do Q")
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    for name, folder in (("DOCS_DIR", "docs"), ("PREVIEWS_DIR", "previews"), ("CROPS_DIR", "crops")):
        path = tmp_path / folder
        path.mkdir()
        for module in (cfg, ingest, ast_module, main):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, path)
    monkeypatch.setattr(indexing, "CACHE_DIR", tmp_path / "cache")
    calls = []

    def embed(texts):
        calls.append(list(texts))
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result

    monkeypatch.setattr(indexing, "embed_texts", embed)
    monkeypatch.setattr(main, "get_embedding_model", lambda: object())
    monkeypatch.setattr(main, "DOCUMENT_VAULT", {})
    monkeypatch.setattr(main, "DOCUMENT_ASTS", {})
    monkeypatch.setattr(main, "ACTIVE_DOC_ID", "")
    monkeypatch.setattr(main, "ACTIVE_RETRIEVER", None)
    monkeypatch.setattr(main, "ACTIVE_CHUNKS", [])
    return calls


def test_disk_cache_skips_processing_and_preserves_renamed_citations(isolated, monkeypatch):
    first = indexing.prepare_document(TABLE, "contract.txt")
    assert not first["cache_hit"]
    assert first["cross_modal_links"] > 0
    monkeypatch.setattr(indexing, "build_document_ast", lambda *a, **k: pytest.fail("Cache hit rebuilt AST"))
    monkeypatch.setattr(indexing, "extract_pages_from_text", lambda *a: pytest.fail("Cache hit extracted text"))
    second = indexing.prepare_document(TABLE, "renamed.md")
    assert second["cache_hit"] and second["doc_id"] == first["doc_id"]
    assert len(isolated) == 1
    assert second["ast"].title == "renamed.md"
    assert all(c["filename"] == "renamed.md" for c in second["chunks"])
    assert second["ast"].get_cross_modal_links() == first["ast"].get_cross_modal_links()
    np.testing.assert_array_equal(first["retriever"].embeddings, second["retriever"].embeddings)
    assert second["stage_timings_ms"]["child_embeddings_ms"] == 0
    assert second["stage_timings_ms"]["extraction_ms"] == 0


@pytest.mark.parametrize("change", ["content", "model", "pipeline"])
def test_cache_invalidates_when_inputs_change(isolated, monkeypatch, change):
    first = indexing.prepare_document(TEXT, "same.txt")
    contents = TEXT
    if change == "content":
        contents += b" Retention is seven years."
    elif change == "model":
        monkeypatch.setattr(cfg, "DEFAULT_EMBED_MODEL", "different-model")
    else:
        monkeypatch.setattr(indexing, "PIPELINE_VERSION", "different-version")
    second = indexing.prepare_document(contents, "same.txt")
    assert not second["cache_hit"]
    assert second["doc_id"] != first["doc_id"]
    assert len(isolated) == 2


def test_corrupt_cache_rebuilds_and_write_failure_does_not_break_search(isolated, monkeypatch):
    indexing.prepare_document(TEXT, "policy.txt")
    next(indexing.CACHE_DIR.glob("*.npz")).write_bytes(b"incomplete")
    assert not indexing.prepare_document(TEXT, "policy.txt")["cache_hit"]
    def cannot_save(*args):
        raise OSError("disk full")
    monkeypatch.setattr(indexing, "_save_cache", cannot_save)
    result = indexing.prepare_document(TEXT + b" New policy.", "policy.txt")
    assert result["retriever"].bm25 is not None
    assert result["cache_warning"]


def test_pdf_renders_only_first_preview_and_lazy_crops_match_eager(isolated, monkeypatch):
    pdf = small_pdf()
    render = ingest.pdfium.PdfPage.render
    renders = []
    def counted(page, *args, **kwargs):
        renders.append(kwargs.get("scale"))
        return render(page, *args, **kwargs)
    monkeypatch.setattr(ingest.pdfium.PdfPage, "render", counted)
    prepared = indexing.prepare_document(pdf, "figures.pdf")
    assert renders == [1.5]
    assert len(list(cfg.PREVIEWS_DIR.glob("*.webp"))) == 1
    assert not list(cfg.CROPS_DIR.glob("*.webp"))
    ast = prepared["ast"]
    assert len(ast.get_visual_nodes()) == 2
    assert ast.get_cross_modal_links()
    visual = ast.get_visual_nodes()[0]
    assert visual.image_url == f"/api/crop/{ast.doc_id}/{visual.id}"
    crop = ast_module.get_or_render_visual_crop(ast, visual)
    assert crop.exists() and renders == [1.5, 2.0]
    assert ast_module.get_or_render_visual_crop(ast, visual) == crop
    assert renders == [1.5, 2.0]
    assert ingest.get_or_render_page_image(ast.doc_id, 2).exists()
    assert renders == [1.5, 2.0, 1.5]
    assert ingest.get_or_render_page_image(ast.doc_id, 99) is None
    eager = ast_module.extract_halo_crops("eager", pdf)[0]
    with Image.open(crop) as lazy_image, Image.open(cfg.CROPS_DIR / f"{eager.id}.webp") as eager_image:
        assert lazy_image.size == eager_image.size
        assert lazy_image.tobytes() == eager_image.tobytes()
    assert indexing.prepare_document(pdf, "figures.pdf")["cache_hit"]
    # Losing the retained source invalidates the cache rather than serving broken evidence.
    (cfg.DOCS_DIR / f"{ast.doc_id}.pdf").unlink()
    assert not indexing.prepare_document(pdf, "figures.pdf")["cache_hit"]


def test_pdf_duplicate_table_keeps_resolvable_lazy_url(isolated, monkeypatch):
    duplicate = ast_module.ASTNode("detected", ASTNodeType.VISUAL, "Table 1: Regional Uptime Log", "table", 1,
                                   "root/detected", bbox=(10, 10, 100, 100))
    monkeypatch.setattr(ast_module, "extract_halo_crops", lambda *a, **k: [duplicate])
    ast = ast_module.build_document_ast("table_doc", ingest.extract_pages_from_text(TABLE.decode(), "table.pdf"),
                                        pdf_bytes=b"placeholder", lazy_visuals=True)
    visual = ast.get_visual_nodes()[0]
    assert visual.id != duplicate.id
    assert visual.image_url == f"/api/crop/table_doc/{visual.id}"
    assert visual.bbox == duplicate.bbox


def test_upload_returns_while_background_indexing_is_blocked(isolated, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    class SlowStore:
        def has_document(self, doc_id):
            return False
        def index_ast(self, ast, timings=None):
            entered.set()
            assert release.wait(10), "Test did not release background worker"
            timings.update(leaf_embeddings_ms=20, database_write_ms=10)
    worker = indexing.BackgroundIndexer(SlowStore())
    monkeypatch.setattr(main, "BACKGROUND_INDEXER", worker)
    try:
        with TestClient(main.app) as client:
            with ThreadPoolExecutor() as requests:
                pending = requests.submit(client.post, "/upload", files={"file": ("policy.txt", TEXT)})
                assert entered.wait(5)
                # The endpoint must finish even though the secondary index cannot finish.
                response = pending.result(timeout=2)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["attachment_duration_ms"] >= data["indexing_duration_ms"]
            assert data["background_indexing"]["status"] in ("queued", "indexing")
            assert main.ACTIVE_RETRIEVER.bm25 is not None
            assert len(isolated) == 1
            next_upload = client.post("/upload", files={"file": ("other.txt", TEXT + b" Retention is seven years.")})
            assert next_upload.status_code == 200
            next_doc = next_upload.json()["doc_id"]
            assert main.ACTIVE_DOC_ID == next_doc
            release.set()
            worker.shutdown()
            assert main.ACTIVE_DOC_ID == next_doc
            status = client.get("/vault").json()["vault"][0]["background_indexing"]
            assert status["status"] == "ready"
            assert status["stage_timings_ms"] == {"leaf_embeddings_ms": 20, "database_write_ms": 10}
    finally:
        release.set()
        worker.shutdown()


def test_background_failure_visible_and_retryable(isolated, monkeypatch):
    class UnreliableStore:
        attempts = 0
        def has_document(self, doc_id):
            return False
        def index_ast(self, ast, timings=None):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("unavailable")
    worker = indexing.BackgroundIndexer(UnreliableStore())
    monkeypatch.setattr(main, "BACKGROUND_INDEXER", worker)
    with TestClient(main.app) as client:
        assert client.post("/upload", files={"file": ("policy.txt", TEXT)}).status_code == 200
        worker.shutdown()
        state = client.get("/vault").json()["vault"][0]["background_indexing"]
        assert state["status"] == "failed" and state["error"]
        assert main.ACTIVE_RETRIEVER.bm25 is not None
        retry = client.post("/upload", files={"file": ("policy.txt", TEXT)})
        assert retry.json()["cache_hit"]
        worker.shutdown()
        assert client.get("/vault").json()["vault"][0]["background_indexing"]["status"] == "ready"


def test_preexisting_lance_table_is_reused():
    class ExistingStore:
        def has_document(self, doc_id):
            return True
        def index_ast(self, *args, **kwargs):
            pytest.fail("Existing table should not be embedded again")
    worker = indexing.BackgroundIndexer(ExistingStore())
    state = worker.submit(DocumentAST("existing"))
    worker.shutdown()
    assert state["status"] == "ready" and state["reused"]


def test_background_lance_persistence_and_real_stage_measurements(isolated, monkeypatch, tmp_path):
    import app.vector_store as vector_store
    prepared = indexing.prepare_document(TABLE, "table.txt")
    monkeypatch.setattr(vector_store, "embed_texts", indexing.embed_texts)
    store = vector_store.LanceVectorStore(tmp_path / "lance")
    worker = indexing.BackgroundIndexer(store)
    state = worker.submit(prepared["ast"])
    worker.shutdown()
    assert state["status"] == "ready" and not state["reused"]
    assert store.has_document(prepared["doc_id"])
    assert set(state["stage_timings_ms"]) == {"leaf_embeddings_ms", "database_write_ms"}
    assert all(value >= 0 for value in state["stage_timings_ms"].values())
    assert len(isolated) == 2  # Search sentences and AST leaves are embedded separately.
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    results = store.search(prepared["doc_id"], vector, ast=prepared["ast"])
    assert results and any(r.get("linked_visuals") for r in results)


def test_text_crop_endpoint_survives_disk_cache_restore(isolated, monkeypatch):
    class ExistingStore:
        def has_document(self, doc_id):
            return True
    monkeypatch.setattr(main, "BACKGROUND_INDEXER", indexing.BackgroundIndexer(ExistingStore()))
    with TestClient(main.app) as client:
        first = client.post("/upload", files={"file": ("table.md", TABLE)}).json()
        visual_url = main.DOCUMENT_ASTS[first["doc_id"]].get_visual_nodes()[0].image_url
        main.DOCUMENT_ASTS.clear()
        main.DOCUMENT_VAULT.clear()
        # A saved citation must work even before the document is attached again.
        assert client.get(visual_url).status_code == 200
        assert not main.DOCUMENT_ASTS and not main.DOCUMENT_VAULT
        for path in cfg.CROPS_DIR.glob("*.webp"):
            path.unlink()
        restored = client.post("/upload", files={"file": ("table.md", TABLE)}).json()
        assert restored["cache_hit"]
        assert restored["doc_id"] == first["doc_id"]
        ast = client.get("/api/ast").json()
        visual = next(n for n in ast["nodes"].values() if n["type"] == "visual")
        assert not list(cfg.CROPS_DIR.glob("*.webp"))
        response = client.get(visual["image_url"])
        assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
        assert response.content[:4] == b"RIFF"
        assert client.get(visual["image_url"]).content == response.content
        assert client.get(f"/api/crop/{first['doc_id']}/unknown").status_code == 404
        assert client.get("/api/crop/missing/unknown").status_code == 404
        assert client.get("/api/discrepancies").json()["verdict"] == "FLAGGED"


def test_startup_preloads_before_serving_and_handles_failure(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "get_embedding_model", lambda: calls.append("loaded"))
    with TestClient(main.app) as client:
        assert calls == ["loaded"]
        assert client.get("/health").json()["embedding_preload_succeeded"]
    def unavailable():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(main, "get_embedding_model", unavailable)
    with TestClient(main.app) as client:
        assert client.get("/health").json()["embedding_preload_succeeded"] is False


def test_concurrent_model_initialization_uses_one_instance(monkeypatch):
    import fastembed
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return object()
    monkeypatch.setattr(embeddings, "_embedding_model", None)
    monkeypatch.setattr(fastembed, "TextEmbedding", create)
    with ThreadPoolExecutor(max_workers=4) as pool:
        models = list(pool.map(lambda _: embeddings.get_embedding_model(), range(8)))
    assert len(calls) == 1 and all(model is models[0] for model in models)
