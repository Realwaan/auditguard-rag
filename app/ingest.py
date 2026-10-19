import io
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
import pypdfium2 as pdfium
from app.config import PREVIEWS_DIR
from app.classifier import classify_document

def render_page_previews(pdf_bytes: bytes, doc_prefix: str) -> Dict[int, str]:
    """
    Renders every page of a PDF to high-resolution web PNGs using pypdfium2.
    Returns a mapping: page_number -> web_image_url
    """
    page_map = {}
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        for idx, page in enumerate(pdf):
            page_num = idx + 1
            image = page.render(scale=1.5).to_pil()
            file_name = f"{doc_prefix}_page_{page_num}.png"
            file_path = PREVIEWS_DIR / file_name
            image.save(file_path, "PNG")
            page_map[page_num] = f"/static/previews/{file_name}"
    except Exception as e:
        print(f"Warning: Could not render page preview with pypdfium2: {e}")
    return page_map

def extract_pages_from_pdf(pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract page-level text and metadata from raw PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "filename": filename,
                "page": idx + 1,
                "text": text
            })
    return pages

def extract_pages_from_text(raw_text: str, filename: str) -> List[Dict[str, Any]]:
    """Fallback for raw .txt or .md files."""
    chunks = [c.strip() for c in raw_text.split("\n\n\n") if c.strip()]
    if not chunks:
        chunks = [raw_text]
    pages = []
    for idx, c in enumerate(chunks, 1):
        pages.append({
            "filename": filename,
            "page": idx,
            "text": c
        })
    return pages

def hierarchical_chunk(pages: List[Dict[str, Any]], page_images: Dict[int, str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Implements Small-to-Big (Parent-Child) chunking.
    Child sentences are indexed for precise vector matching.
    Parent paragraphs are passed to Gemma 4 for rich, coherent context.
    Also runs automatic document type classification.
    """
    page_images = page_images or {}
    chunks = []
    chunk_id = 0

    # Auto-classify based on the first 1-2 pages
    combined_header = " ".join([p["text"][:1000] for p in pages[:2]])
    classification = classify_document(combined_header)

    for page in pages:
        page_num = page["page"]
        img_url = page_images.get(page_num)

        # Split page into logical paragraphs
        paragraphs = [p.strip() for p in page["text"].split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [page["text"].strip()]

        for p_idx, para in enumerate(paragraphs):
            # Split paragraph into child sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', para) if len(s.strip()) > 15]
            if not sentences:
                sentences = [para]

            for sent in sentences:
                chunks.append({
                    "id": chunk_id,
                    "filename": page["filename"],
                    "page": page_num,
                    "parent_text": para,
                    "child_text": sent,
                    "image_url": img_url,
                    "has_visual": bool(img_url)
                })
                chunk_id += 1

    return chunks, classification
