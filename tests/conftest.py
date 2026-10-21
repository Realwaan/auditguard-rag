"""Keep automated checks offline and away from the user's document vault."""
import os
from pathlib import Path
import tempfile

import pytest


def pytest_configure(config):
    import app.config as settings
    # Explicit test-owned storage, configured before app.main opens LanceDB.
    root = Path(tempfile.mkdtemp(prefix="auditguard-tests-", dir=os.environ.get("TEMP")))
    for name, subdir in (("DOCS_DIR", "docs"), ("INDEX_DIR", "index"),
                         ("PREVIEWS_DIR", "previews"), ("CROPS_DIR", "crops")):
        path = root / subdir
        path.mkdir()
        setattr(settings, name, path)


@pytest.fixture(autouse=True)
def offline_providers(monkeypatch):
    import app.generator as generator
    import app.llm as llm

    def offline():
        raise RuntimeError("Live inference is disabled during automated tests")

    monkeypatch.setattr(generator, "get_genai_client", lambda: None)
    monkeypatch.setattr(generator, "get_openai_client", offline)
    monkeypatch.setattr(llm, "get_genai_client", lambda: None)
    monkeypatch.setattr(llm, "get_openai_client", offline)
    # Discrepancy tests must not call a live model either.
    import sys
    if "app.main" in sys.modules:
        monkeypatch.setattr(sys.modules["app.main"], "USE_GOOGLE_GENAI", False)


@pytest.fixture(autouse=True)
def local_embeddings(monkeypatch):
    import app.embeddings as embeddings

    def cached_model():
        if embeddings._embedding_model is None:
            from fastembed import TextEmbedding
            cache = os.environ.get("FASTEMBED_CACHE_PATH")
            if not cache:
                cache = str(Path(__file__).resolve().parents[1] / "hackathon-setup" / "model-cache")
            embeddings._embedding_model = TextEmbedding(
                model_name=embeddings.DEFAULT_EMBED_MODEL, cache_dir=cache,
                local_files_only=True, threads=2,
            )
        return embeddings._embedding_model

    monkeypatch.setattr(embeddings, "get_embedding_model", cached_model)
