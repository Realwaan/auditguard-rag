import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader

from tests.test_api_regressions import client
from tests.test_answer_regressions import evidence
import app.main as main
import app.generator as generator
import app.ingest as ingest
import app.llm as llm


@pytest.mark.parametrize('mode', ['empty', 'exception', 'competing_pages'])
def test_no_ocr_pdf_retrieval_and_model_image_payload(client, monkeypatch, mode):
    images, page_texts = [], []
    for i in range(7 if mode == 'competing_pages' else 1):
        text = ('Standard support response time is 4 hours.' if i == 0 else
                f'Support response time policy for tier {i + 1}. The support response time value is not specified here.')
        img = Image.new('RGB', (800, 300), 'white')
        ImageDraw.Draw(img).text((30, 70), text, fill='black')
        images.append(img)
        page_texts.append(text)
    buf = io.BytesIO()
    images[0].save(buf, format='PDF', save_all=True, append_images=images[1:])
    pdf = buf.getvalue()
    assert all(not (p.extract_text() or '').strip() for p in PdfReader(io.BytesIO(pdf)).pages)
    ocr_calls = []
    def ocr(image):
        idx = len(ocr_calls)
        ocr_calls.append(idx)
        if mode == 'exception':
            raise RuntimeError('OCR unavailable')
        return page_texts[idx] if idx > 0 else ''
    monkeypatch.setattr(ingest, 'ocr_extract_page_image', ocr)
    upload = client.post('/upload', files={'file': ('qa_scan.pdf', pdf, 'application/pdf')})
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()['doc_id']
    chunks = main.DOCUMENT_VAULT[doc_id]['chunks']
    scan = next(c for c in chunks if c['page'] == 1)
    assert scan['is_scanned'] is True and scan['is_ocr'] is False
    image_bytes, mime = llm.load_image_bytes(scan['image_url'])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        # Derive the citation from the real request, not from a prearranged retriever stub.
        import re
        prompt = kwargs['messages'][1]['content'][0]['text']
        cid = re.search(r'\[(\d+)\] \(File: qa_scan.pdf, Page 1\)', prompt)[1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f'Standard support response time is 4 hours [{cid}].'))], usage=None)
    monkeypatch.setattr(generator, 'USE_GOOGLE_GENAI', False)
    monkeypatch.setattr(generator, 'get_openai_client', lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    response = client.post('/ask', json={'query': 'What is the support response time?', 'doc_id': doc_id, 'include_baseline': False})
    assert response.status_code == 200
    data = response.json()
    assert any(c['page'] == 1 for c in data['sources'])
    assert data['evaluation']['evaluation_method'] == 'multimodal_visual_inspection'
    assert data['has_multimodal_context'] is True
    parts = calls[0]['messages'][1]['content']
    expected = f'data:{mime};base64,' + base64.b64encode(image_bytes).decode()
    image_index = next(i for i, p in enumerate(parts) if p.get('type') == 'image_url' and p['image_url']['url'] == expected)
    assert 'source page: 1' in parts[image_index - 1]['text']
    assert data['citation_verification']['all_supported'] is False
    assert data['status'] == 'CAUTION_UNSUPPORTED_CITATION'


def test_lancedb_failure_preserves_previous_vault(client, monkeypatch):
    assert client.post('/upload', files={'file': ('first.txt', b'Support response time is 4 hours.')}).status_code == 200
    previous_id = main.ACTIVE_DOC_ID
    previous_retriever = main.ACTIVE_RETRIEVER
    previous_vault = dict(main.DOCUMENT_VAULT)
    def fail(*args, **kwargs):
        raise RuntimeError('Disk full')
    monkeypatch.setattr(main.LANCE_STORE, 'index_ast', fail)
    response = client.post('/upload', files={'file': ('second.txt', b'Support response time is 8 hours.')})
    assert response.status_code == 503
    assert main.ACTIVE_DOC_ID == previous_id
    assert main.ACTIVE_RETRIEVER is previous_retriever
    assert main.DOCUMENT_VAULT == previous_vault


def test_invalid_model_citation_is_not_silently_erased(monkeypatch):
    monkeypatch.setattr(generator, 'USE_GOOGLE_GENAI', False)
    result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Uptime is 99.9% [1][9].'))], usage=None)
    monkeypatch.setattr(generator, 'get_openai_client', lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: result))))
    answer = generator.generate_grounded_answer('What is the uptime?', [evidence('Uptime is 99.9%.')])
    assert '[9]' in answer['answer']
    assert answer['citation_verification']['verdict'] == 'INVALID_CITATION'


def test_stream_failover_uses_each_providers_model(monkeypatch):
    monkeypatch.setattr(generator, 'USE_GOOGLE_GENAI', True)
    monkeypatch.setattr(generator, 'get_default_model', lambda: 'google-model')
    monkeypatch.setattr(generator, 'LLM_MODEL', 'venue-model')
    def fail(**kwargs):
        raise RuntimeError('Google unavailable')
    monkeypatch.setattr(generator, 'get_genai_client', lambda: SimpleNamespace(models=SimpleNamespace(generate_content_stream=fail)))
    models = []
    def create(**kwargs):
        models.append(kwargs['model'])
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='Uptime is 99.9% [1].'))])])
    monkeypatch.setattr(generator, 'get_openai_client', lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert ''.join(generator.stream_grounded_answer('What is uptime?', [evidence('Uptime is 99.9%.')])) == 'Uptime is 99.9% [1].'
    assert models == ['venue-model']
