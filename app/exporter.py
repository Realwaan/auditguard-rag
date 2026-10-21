"""
Multi-Format Report Exporter for AuditGuard RAG.
Generates genuine PDF (.pdf), Microsoft Word OpenXML (.docx), CSV (.csv),
Markdown (.md), and JSON (.json) files with zero external binary dependencies.
"""

from __future__ import annotations
import csv
import io
import textwrap
import zipfile
from xml.sax.saxutils import escape as xml_escape
from typing import List, Dict, Any


def _pdf_escape(text: str) -> str:
    """Escapes text for PDF literal strings using Latin-1 safe characters."""
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_bytes(
    title: str,
    timestamp: str,
    doc_label: str,
    confidence: float,
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> bytes:
    """
    Constructs a valid multi-page PDF 1.4 document using standard Type1 Helvetica fonts.
    All text is selectable vector text.
    """
    lines_to_render: List[tuple[str, int, bool]] = []  # (text, font_size, is_bold)

    lines_to_render.append((title, 15, True))
    lines_to_render.append((f"Generated: {timestamp} | Classification: {doc_label} | Grounding: {confidence}%", 9, False))
    lines_to_render.append(("", 8, False))

    lines_to_render.append(("1. Question", 12, True))
    for wrapped in textwrap.wrap(query.strip(), width=88) or [""]:
        lines_to_render.append((wrapped, 10, False))
    lines_to_render.append(("", 8, False))

    lines_to_render.append(("2. Verified Answer", 12, True))
    for raw_line in answer.splitlines():
        clean_line = raw_line.strip()
        if not clean_line:
            lines_to_render.append(("", 6, False))
            continue
        is_heading = clean_line.startswith("#")
        clean_line = clean_line.lstrip("#").strip()
        for wrapped in textwrap.wrap(clean_line, width=88) or [""]:
            lines_to_render.append((wrapped, 10, is_heading))
    lines_to_render.append(("", 8, False))

    if sources:
        lines_to_render.append(("3. Cited Document Sources", 12, True))
        for s in sources:
            cid = s.get("citation_id", 1)
            doc = s.get("filename", "Document")
            page = s.get("page", 1)
            score = round(float(s.get("dense_score", 0.0)) * 100, 1)
            header = f"[{cid}] {doc} (Page {page} - {score}% Match)"
            lines_to_render.append((header, 10, True))
            excerpt = s.get("parent_text", s.get("child_text", "")).replace("\n", " ").strip()
            for wrapped in textwrap.wrap(excerpt, width=88)[:6]:
                lines_to_render.append((f"    {wrapped}", 9, False))
            lines_to_render.append(("", 5, False))

    # Paginate lines into 612 x 792 pt pages
    pages_streams: List[str] = []
    y_cursor = 740
    current_ops: List[str] = []

    for text, size, is_bold in lines_to_render:
        line_height = size + 5
        if y_cursor - line_height < 52:
            pages_streams.append("\n".join(current_ops))
            current_ops = []
            y_cursor = 740

        if text:
            font_ref = "/F2" if is_bold else "/F1"
            escaped = _pdf_escape(text)
            current_ops.append(f"BT {font_ref} {size} Tf 50 {y_cursor} Td ({escaped}) Tj ET")
        y_cursor -= line_height

    if current_ops or not pages_streams:
        pages_streams.append("\n".join(current_ops))

    # Assemble PDF object table
    # Object 1: Catalog, Object 2: Pages, Object 3: Font F1 (Helvetica), Object 4: Font F2 (Helvetica-Bold)
    objects: Dict[int, bytes] = {}
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

    page_obj_ids: List[int] = []
    next_id = 5
    for stream_str in pages_streams:
        stream_bytes = stream_str.encode("latin-1", errors="replace")
        content_id = next_id
        page_id = next_id + 1
        next_id += 2

        objects[content_id] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        page_obj_ids.append(page_id)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>".encode("ascii")
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: Dict[int, int] = {}
    max_obj_id = max(objects.keys())

    for oid in range(1, max_obj_id + 1):
        offsets[oid] = out.tell()
        out.write(f"{oid} 0 obj\n".encode("ascii"))
        out.write(objects[oid])
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(f"xref\n0 {max_obj_id + 1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for oid in range(1, max_obj_id + 1):
        out.write(f"{offsets[oid]:010d} 00000 n \n".encode("ascii"))

    out.write(
        f"trailer\n<< /Size {max_obj_id + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return out.getvalue()


def build_docx_bytes(
    title: str,
    timestamp: str,
    doc_label: str,
    confidence: float,
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> bytes:
    """
    Constructs a valid Microsoft Word OpenXML (.docx) archive using standard zipfile.
    Opens natively in Microsoft Word, Google Docs, and LibreOffice.
    """
    def make_para(text: str, bold: bool = False, size_half_pt: int = 22) -> str:
        escaped = xml_escape(text)
        b_tag = "<w:b/>" if bold else ""
        return (
            f"<w:p><w:r><w:rPr>{b_tag}<w:sz w:val=\"{size_half_pt}\"/></w:rPr>"
            f"<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"
        )

    paragraphs: List[str] = [
        make_para(title, bold=True, size_half_pt=32),
        make_para(f"Generated: {timestamp} | Classification: {doc_label} | Confidence: {confidence}%", bold=False, size_half_pt=18),
        make_para(""),
        make_para("1. Question", bold=True, size_half_pt=26),
        make_para(query, bold=False, size_half_pt=22),
        make_para(""),
        make_para("2. Verified Finding", bold=True, size_half_pt=26),
    ]

    for line in answer.splitlines():
        clean = line.strip()
        is_hdr = clean.startswith("#")
        paragraphs.append(make_para(clean.lstrip("#").strip(), bold=is_hdr, size_half_pt=24 if is_hdr else 22))

    if sources:
        paragraphs.append(make_para(""))
        paragraphs.append(make_para("3. Cited Document Sources", bold=True, size_half_pt=26))
        for s in sources:
            cid = s.get("citation_id", 1)
            doc = s.get("filename", "Document")
            page = s.get("page", 1)
            score = round(float(s.get("dense_score", 0.0)) * 100, 1)
            excerpt = s.get("parent_text", s.get("child_text", "")).replace("\n", " ").strip()
            paragraphs.append(make_para(f"[{cid}] {doc} (Page {page} — {score}% Match)", bold=True, size_half_pt=21))
            paragraphs.append(make_para(excerpt, bold=False, size_half_pt=20))

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body>"
        + "".join(paragraphs)
        + "<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr>"
        "</w:body></w:document>"
    )

    content_types_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "</Types>"
    )

    rels_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "</Relationships>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def build_csv_string(
    timestamp: str,
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> str:
    """Builds a UTF-8 CSV report containing the finding and all cited source rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Generated At", "Question", "Verified Answer", "Citation ID", "Document", "Page", "Similarity (%)", "Source Excerpt"])
    if not sources:
        writer.writerow([timestamp, query, answer, "", "", "", "", ""])
    else:
        for idx, s in enumerate(sources):
            cid = f"[{s.get('citation_id', idx + 1)}]"
            doc = s.get("filename", "Document")
            page = s.get("page", 1)
            score = round(float(s.get("dense_score", 0.0)) * 100, 1)
            excerpt = s.get("parent_text", s.get("child_text", "")).replace("\n", " ").strip()
            writer.writerow([
                timestamp if idx == 0 else "",
                query if idx == 0 else "",
                answer if idx == 0 else "",
                cid,
                doc,
                page,
                score,
                excerpt,
            ])
    return buf.getvalue()
