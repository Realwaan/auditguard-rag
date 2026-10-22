"""
Compiler-Grade Document Abstract Syntax Tree (AST) Builder & Cross-Modal Symbol Linker.
Constructs a 3-tier Document AST (Root -> Section -> Paragraph <-> Visual Leaf),
executes Collision-Bounded Halo Cropping via pypdfium2 + Pillow,
and wires deterministic 3-tier cross-modal sibling links.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw
import pypdfium2 as pdfium

from app.config import CROPS_DIR, DOCS_DIR
from app.pdf_lock import serialized_pdf


class ASTNodeType(str, Enum):
    ROOT = "root"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    VISUAL = "visual"


@dataclass
class ASTNode:
    id: str
    type: ASTNodeType
    title: str
    text: str
    page: int
    materialized_path: str  # e.g., "root/01_executive_summary/p_1"
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1) in pt
    image_url: Optional[str] = None
    visual_type: Optional[str] = None  # "figure", "table", "chart", "diagram"
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    linked_node_ids: List[str] = field(default_factory=list)  # Cross-modal sibling links
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value if isinstance(self.type, ASTNodeType) else str(self.type)
        return d


class DocumentAST:
    """In-memory Abstract Syntax Tree for structured multimodal documents."""

    def __init__(self, doc_id: str, title: str = "Document"):
        self.doc_id = doc_id
        self.title = title
        self.root_id = f"{doc_id}_root"
        self.nodes: Dict[str, ASTNode] = {}

        # Initialize root node
        root_node = ASTNode(
            id=self.root_id,
            type=ASTNodeType.ROOT,
            title=title,
            text=f"Document Root: {title}",
            page=1,
            materialized_path="root",
        )
        self.nodes[self.root_id] = root_node

    def add_node(self, node: ASTNode, parent_id: Optional[str] = None) -> None:
        """Add a node and attach to parent if specified."""
        self.nodes[node.id] = node
        p_id = parent_id or node.parent_id or self.root_id
        if p_id in self.nodes and node.id != p_id:
            node.parent_id = p_id
            if node.id not in self.nodes[p_id].children_ids:
                self.nodes[p_id].children_ids.append(node.id)

    def get_node(self, node_id: str) -> Optional[ASTNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[ASTNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> List[ASTNode]:
        ancestors = []
        curr = self.nodes.get(node_id)
        visited = set()
        while curr and curr.parent_id and curr.parent_id not in visited:
            visited.add(curr.parent_id)
            parent = self.nodes.get(curr.parent_id)
            if parent:
                ancestors.append(parent)
                curr = parent
            else:
                break
        return ancestors

    def get_leaves(self) -> List[ASTNode]:
        """Returns all leaf nodes (paragraphs and visuals)."""
        return [
            n for n in self.nodes.values()
            if n.type in (ASTNodeType.PARAGRAPH, ASTNodeType.VISUAL)
        ]

    def get_visual_nodes(self) -> List[ASTNode]:
        return [n for n in self.nodes.values() if n.type == ASTNodeType.VISUAL]

    def get_cross_modal_links(self) -> List[Dict[str, Any]]:
        """Returns all bidirectional edges between text and visuals."""
        links = []
        seen = set()
        for node in self.nodes.values():
            for target_id in node.linked_node_ids:
                edge_key = tuple(sorted([node.id, target_id]))
                if edge_key not in seen and target_id in self.nodes:
                    seen.add(edge_key)
                    target = self.nodes[target_id]
                    edge_meta = node.metadata.get("edge_meta", {}).get(target_id, {})
                    tier = edge_meta.get("tier") or node.metadata.get("link_tier", 1)
                    weight = edge_meta.get("weight") or node.metadata.get("link_weight", 1.0)
                    links.append({
                        "source_id": node.id,
                        "source_type": node.type.value,
                        "target_id": target.id,
                        "target_type": target.type.value,
                        "tier": tier,
                        "weight": weight,
                    })
        return links

    def get_all_pages(self) -> List[int]:
        """Returns sorted list of all unique page numbers in AST."""
        return sorted(list({n.page for n in self.nodes.values() if n.page}))

    def get_page_text(self, page: int) -> str:
        """Aggregates all paragraph and section text on a specific page."""
        page_nodes = [
            n for n in self.nodes.values()
            if n.page == page and n.text and n.type == ASTNodeType.PARAGRAPH
        ]
        if not page_nodes:
            page_nodes = [
                n for n in self.nodes.values()
                if n.page == page and n.text and n.type != ASTNodeType.ROOT
            ]
        return "\n\n".join(n.text.strip() for n in page_nodes if n.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Full serializable dictionary for API responses & UI graph rendering."""
        links = self.get_cross_modal_links()
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "filename": self.title,
            "root_id": self.root_id,
            "total_nodes": len(self.nodes),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "cross_modal_links": links,
            "cross_modal_links_count": len(links),
        }


def render_text_visual_crop(vis_id: str, title: str, body_text: str) -> str:
    """
    Renders a structured text table or flowchart block into a lossless WebP visual crop
    so text/markdown documents have genuine, inspectable visual artifacts on disk.
    """
    lines = [title] + [ln.strip() for ln in body_text.splitlines() if ln.strip() and ln.strip() != title.strip()]
    max_chars = max((len(ln) for ln in lines), default=40)
    img_w = max(560, min(1100, max_chars * 8 + 48))
    img_h = max(160, min(800, len(lines) * 26 + 44))

    img = Image.new("RGB", (img_w, img_h), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (img_w, 36)], fill=(16, 185, 129))
    draw.text((16, 10), title[:110], fill=(2, 6, 23))

    y = 48
    for idx, ln in enumerate(lines[1:]):
        row_bg = (30, 41, 59) if idx % 2 == 0 else (15, 23, 42)
        draw.rectangle([(10, y - 3), (img_w - 10, y + 20)], fill=row_bg)
        text_color = (110, 231, 183) if idx == 0 and "|" in ln else (226, 232, 240)
        draw.text((18, y), ln[:125], fill=text_color)
        y += 24
        if y > img_h - 24:
            break

    crop_file_name = f"{vis_id}.webp"
    crop_path = CROPS_DIR / crop_file_name
    img.save(crop_path, "WEBP", quality=85)
    return f"/static/crops/{crop_file_name}"


def _locate_text_bbox_on_page(
    text_page: Any,
    search_snippet: str,
    width_pt: float,
    height_pt: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Uses pypdfium2 textpage search to locate real PDF coordinates for a text/table block."""
    query_str = search_snippet.strip()[:60]
    if not query_str:
        return None
    try:
        searcher = text_page.search(query_str, match_case=False, match_whole_word=False)
        first_match = searcher.get_next()
        if first_match:
            char_index, char_count = first_match
            rect_count = text_page.count_rects(char_index, char_count)
            rects = [text_page.get_rect(i) for i in range(rect_count)]
            if rects:
                x0 = min(r[0] for r in rects)
                y0 = min(r[1] for r in rects)
                x1 = max(r[2] for r in rects)
                y1 = max(r[3] for r in rects)
                # Expand downward to encompass multi-line table body (~120pt)
                return (max(18.0, x0 - 12.0), max(18.0, y0 - 120.0), min(width_pt - 18.0, max(x1 + 12.0, width_pt - 36.0)), min(height_pt - 18.0, y1 + 12.0))
    except Exception:
        pass
    return None


@serialized_pdf
def extract_halo_crops(
    doc_id: str,
    pdf_bytes: bytes,
    detected_boxes_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    render_crops: bool = True,
) -> List[ASTNode]:
    """
    Collision-Bounded Halo Cropper:
    1. Uses pypdfium2 to inspect page image objects and extract surrounding caption/text context.
    2. Expands bounding boxes by 12% halo padding within page margins.
    3. Locates embedded tables and extracts bounded text from the PDF textpage.
    4. Renders and saves WebP crops in app/static/crops/.
    """
    visual_nodes: List[ASTNode] = []
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except Exception as e:
        print(f"HaloCropper: Could not open PDF bytes: {e}")
        return visual_nodes

    try:
        for page_idx in range(len(pdf)):
            page_num = page_idx + 1
            page = pdf[page_idx]
            width_pt, height_pt = page.get_width(), page.get_height()
            text_page = page.get_textpage()
            page_text = text_page.get_text_range() or ""

            candidate_boxes: List[Dict[str, Any]] = []
            if detected_boxes_by_page and page_num in detected_boxes_by_page:
                candidate_boxes.extend(detected_boxes_by_page[page_num])
            else:
                try:
                    img_idx = 0
                    for obj in page.get_objects():
                        # PDF object type 3 corresponds to Image
                        if getattr(obj, "type", None) == 3:
                            # Newer PDFium helpers renamed get_pos() to get_bounds().
                            get_bounds = getattr(obj, "get_bounds", None) or obj.get_pos
                            bbox = get_bounds()  # (left, bottom, right, top) in PDF coordinates
                            if (bbox[2] - bbox[0] > 40) and (bbox[3] - bbox[1] > 40):
                                img_idx += 1
                                # Inspect nearby caption region (45pt below and above image)
                                cap_x0 = max(0.0, bbox[0] - 20.0)
                                cap_y0 = max(0.0, bbox[1] - 45.0)
                                cap_x1 = min(width_pt, bbox[2] + 20.0)
                                cap_y1 = min(height_pt, bbox[3] + 45.0)
                                nearby_text = ""
                                try:
                                    nearby_text = text_page.get_text_bounded(cap_x0, cap_y0, cap_x1, cap_y1).strip()
                                except Exception:
                                    pass

                                cap_match = re.search(
                                    r'((?:Figure|Fig\.?|Table|Chart|Exhibit|Diagram)\s+[0-9IVX]+[a-zA-Z]?(?:[:.\-\s][^\n]+)?)',
                                    nearby_text or page_text,
                                    re.IGNORECASE,
                                )
                                title = cap_match.group(1).strip() if cap_match else f"Visual Figure #{img_idx} [P{page_num}]"
                                candidate_boxes.append({
                                    "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                                    "title": title,
                                    "text": nearby_text if nearby_text else title,
                                    "visual_type": "figure",
                                    "has_explicit_caption": bool(cap_match),
                                })
                except Exception as obj_err:
                    print(f"Object detection on page {page_num} warning: {obj_err}")

            # Check for structured tables in page text
            table_matches = re.finditer(
                r'(?:Table\s+(\d+[a-zA-Z]?)[^\n]*\n)(?:(?:\|[^\n]+\|)\n?)+',
                page_text,
                re.IGNORECASE,
            )
            for m in table_matches:
                tbl_block = m.group(0).strip()
                tbl_title = tbl_block.splitlines()[0].strip()
                real_bbox = _locate_text_bbox_on_page(text_page, tbl_title, width_pt, height_pt)
                candidate_boxes.append({
                    "bbox": real_bbox or (36.0, height_pt * 0.25, width_pt - 36.0, height_pt * 0.75),
                    "title": tbl_title,
                    "text": tbl_block,
                    "visual_type": "table",
                    "has_explicit_caption": True,
                })

            pil_page = None
            for v_idx, cand in enumerate(candidate_boxes, 1):
                vis_id = f"{doc_id}_vis_p{page_num}_{v_idx}"
                bbox = cand.get("bbox", (50.0, 100.0, width_pt - 50.0, height_pt - 100.0))

                raw_x0, raw_y0, raw_x1, raw_y1 = bbox
                x0, x1 = min(raw_x0, raw_x1), max(raw_x0, raw_x1)
                y0, y1 = min(raw_y0, raw_y1), max(raw_y0, raw_y1)

                w = max(1.0, x1 - x0)
                h = max(1.0, y1 - y0)

                # 12% Halo Expansion bounded by page dimensions
                halo_x = w * 0.12
                halo_y = h * 0.12
                crop_x0 = max(0.0, x0 - halo_x)
                crop_x1 = min(width_pt, x1 + halo_x)
                crop_y0 = max(0.0, y0 - halo_y)
                crop_y1 = min(height_pt, y1 + halo_y)

                crop_file_name = f"{vis_id}.webp"
                if render_crops:
                    if pil_page is None:
                        pil_page = page.render(scale=2.0).to_pil()
                    img_w, img_h = pil_page.size

                    calc_x0 = int((crop_x0 / width_pt) * img_w)
                    calc_x1 = int((crop_x1 / width_pt) * img_w)
                    calc_y0 = int(((height_pt - crop_y1) / height_pt) * img_h)
                    calc_y1 = int(((height_pt - crop_y0) / height_pt) * img_h)

                    img_x0 = max(0, min(img_w - 2, min(calc_x0, calc_x1)))
                    img_x1 = max(img_x0 + 1, min(img_w, max(calc_x0, calc_x1)))
                    img_y0 = max(0, min(img_h - 2, min(calc_y0, calc_y1)))
                    img_y1 = max(img_y0 + 1, min(img_h, max(calc_y0, calc_y1)))

                    cropped_img = pil_page.crop((img_x0, img_y0, img_x1, img_y1))
                    crop_file_name = f"{vis_id}.webp"
                    crop_path = CROPS_DIR / crop_file_name
                    cropped_img.save(crop_path, "WEBP", quality=85)

                v_title = cand.get("title") or f"Visual Figure #{v_idx} [P{page_num}]"
                v_text = cand.get("text") or v_title
                node_conf = 0.92 if cand.get("has_explicit_caption") else 0.75

                node = ASTNode(
                    id=vis_id,
                    type=ASTNodeType.VISUAL,
                    title=v_title,
                    text=v_text,
                    page=page_num,
                    materialized_path=f"root/visuals/{vis_id}",
                    bbox=(crop_x0, crop_y0, crop_x1, crop_y1),
                    image_url=(f"/static/crops/{crop_file_name}" if render_crops else f"/api/crop/{doc_id}/{vis_id}"),
                    visual_type=cand.get("visual_type", "figure"),
                    confidence=node_conf,
                )
                visual_nodes.append(node)
    finally:
        pdf.close()

    return visual_nodes


@serialized_pdf
def get_or_render_visual_crop(ast: DocumentAST, node: ASTNode) -> Path | None:
    """Render a previously detected visual without repeating document analysis."""
    if node.type != ASTNodeType.VISUAL:
        return None
    # Only server-generated identifiers may resolve filesystem paths.
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", ast.doc_id) or not re.fullmatch(r"[a-zA-Z0-9_-]+", node.id):
        return None
    path = CROPS_DIR / f"{node.id}.webp"
    if path.exists():
        return path
    if node.bbox:
        pdf_path = DOCS_DIR / f"{ast.doc_id}.pdf"
        if not pdf_path.exists():
            return None
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if not 1 <= node.page <= len(pdf):
                return None
            page = pdf[node.page - 1]
            bitmap = page.render(scale=2.0)
            try:
                image = bitmap.to_pil()
                width, height = image.size
                x0, y0, x1, y1 = node.bbox
                left = max(0, min(width - 2, int(x0 / page.get_width() * width)))
                right = max(left + 1, min(width, int(x1 / page.get_width() * width)))
                top = max(0, min(height - 2, int((page.get_height() - y1) / page.get_height() * height)))
                bottom = max(top + 1, min(height, int((page.get_height() - y0) / page.get_height() * height)))
                image.crop((left, top, right, bottom)).save(path, "WEBP", quality=85)
            finally:
                bitmap.close()
        finally:
            pdf.close()
    else:
        render_text_visual_crop(node.id, node.title, node.text)
    return path


def _wire_bidirectional_link(p_node: ASTNode, v_node: ASTNode, tier: int, weight: float) -> None:
    """Wires a bidirectional cross-modal edge with tier and weight metadata."""
    if v_node.id not in p_node.linked_node_ids:
        p_node.linked_node_ids.append(v_node.id)
    if p_node.id not in v_node.linked_node_ids:
        v_node.linked_node_ids.append(p_node.id)

    # Preserve highest-priority (lowest tier number) on node metadata
    if "link_tier" not in p_node.metadata or tier < p_node.metadata["link_tier"]:
        p_node.metadata["link_tier"] = tier
        p_node.metadata["link_weight"] = weight
    if "link_tier" not in v_node.metadata or tier < v_node.metadata["link_tier"]:
        v_node.metadata["link_tier"] = tier
        v_node.metadata["link_weight"] = weight

    p_node.metadata.setdefault("edge_meta", {})[v_node.id] = {"tier": tier, "weight": weight}
    v_node.metadata.setdefault("edge_meta", {})[p_node.id] = {"tier": tier, "weight": weight}


def link_cross_modal_symbols(ast: DocumentAST) -> None:
    """
    3-Tier Cross-Modal Linker:
    - Tier 1 (Weight 1.0): Exact Symbol Regex ("Figure 3", "Table 1", "Exhibit 4B")
    - Tier 2 (Weight 0.75): Spatial / Same-Section Metric & Adjacency Alignment
    - Tier 3 (Weight 0.50): Parent Section / Page Fallback for unreferenced visuals
    """
    paragraphs = [n for n in ast.nodes.values() if n.type == ASTNodeType.PARAGRAPH]
    visuals = ast.get_visual_nodes()

    if not visuals or not paragraphs:
        return

    symbol_pattern = re.compile(
        r'\b(figure|fig\.?|table|chart|exhibit|diagram)\s*#?([0-9]+[a-z]?(?:\.[0-9]+)?)',
        re.IGNORECASE,
    )

    # Pre-extract symbol tags on visuals (stripping page suffixes like "(Page 3)" or "[P3]")
    visual_symbols: Dict[str, List[Tuple[str, str]]] = {}
    for v_node in visuals:
        clean_v_title = re.sub(r'[\(\[]\s*(?:page|p)\s*\d+\s*[\)\]]', '', v_node.title, flags=re.IGNORECASE)
        v_matches = symbol_pattern.findall(clean_v_title)
        visual_symbols[v_node.id] = [(kind.lower().rstrip('.'), num.lower()) for kind, num in v_matches]

    # 1. Tier 1: Explicit Symbol Matching (e.g. "Figure 3" -> Visual "Figure 3")
    for p_node in paragraphs:
        p_matches = symbol_pattern.findall(p_node.text)
        for p_kind, p_num in p_matches:
            norm_kind = p_kind.lower().rstrip('.')
            if norm_kind == "fig":
                norm_kind = "figure"
            norm_num = p_num.strip().lower()

            for v_node in visuals:
                v_syms = visual_symbols.get(v_node.id, [])
                matched = False
                for v_kind, v_num in v_syms:
                    vk = "figure" if v_kind == "fig" else v_kind
                    if v_num == norm_num and (vk == norm_kind or norm_kind in vk or vk in norm_kind):
                        matched = True
                        break
                if not matched and not v_syms:
                    # Fallback check against clean title if visual title uses non-standard format like "#3"
                    clean_v_title = re.sub(r'[\(\[]\s*(?:page|p)\s*\d+\s*[\)\]]', '', v_node.title, flags=re.IGNORECASE).lower()
                    if f" {norm_num}:" in clean_v_title or f"#{norm_num}" in clean_v_title or clean_v_title.endswith(f" {norm_num}"):
                        matched = True

                if matched:
                    _wire_bidirectional_link(p_node, v_node, tier=1, weight=1.0)

    # 2. Tier 2: Same-Section / Shared-Metric Sibling Alignment
    # Connects visuals to sibling paragraphs in the same section that share quantitative claims (e.g. % or $)
    pct_or_num = re.compile(r'\b\d+(?:\.\d+)?\s*%|\$\d+')
    for v_node in visuals:
        same_sec_paras = [
            p for p in paragraphs
            if (v_node.parent_id and v_node.parent_id != ast.root_id and p.parent_id == v_node.parent_id)
            or (p.page == v_node.page)
        ]
        v_has_metrics = bool(pct_or_num.search(v_node.text))
        for p_node in same_sec_paras:
            if v_node.id in p_node.linked_node_ids:
                continue
            # If both belong to the same non-root section and both contain quantitative metrics, link at Tier 2
            if (
                v_node.parent_id
                and v_node.parent_id != ast.root_id
                and p_node.parent_id == v_node.parent_id
                and v_has_metrics
                and bool(pct_or_num.search(p_node.text))
            ):
                _wire_bidirectional_link(p_node, v_node, tier=2, weight=0.75)

    # 3. Tier 2/3 Spatial Adjacency & Section Fallback for any remaining unlinked visuals
    for v_node in visuals:
        if not v_node.linked_node_ids:
            same_page_paras = [p for p in paragraphs if p.page == v_node.page]
            if same_page_paras:
                target_para = same_page_paras[0]
                _wire_bidirectional_link(target_para, v_node, tier=2, weight=0.75)
            elif v_node.parent_id and v_node.parent_id in ast.nodes:
                sec_paras = [p for p in paragraphs if p.parent_id == v_node.parent_id]
                if sec_paras:
                    _wire_bidirectional_link(sec_paras[0], v_node, tier=3, weight=0.50)


def build_document_ast(
    doc_id: str,
    pages: List[Dict[str, Any]],
    pdf_bytes: Optional[bytes] = None,
    doc_title: Optional[str] = None,
    lazy_visuals: bool = False,
) -> DocumentAST:
    """
    Master Ingestion Function:
    Transforms raw document pages and optional PDF bytes into a complete 3-Tier Document AST.
    Automatically extracts both embedded PDF visuals and Markdown/ASCII tables into VISUAL nodes.
    """
    title = doc_title or doc_id.replace("_", " ").title()
    ast = DocumentAST(doc_id=doc_id, title=title)

    current_section_id = f"{doc_id}_sec_1"
    sec_idx = 1

    # Default Section 1: Overview (removed later if the document immediately opens with its own Section heading)
    first_sec = ASTNode(
        id=current_section_id,
        type=ASTNodeType.SECTION,
        title="1. Document Overview & Scope",
        text="Primary section and scope definitions.",
        page=1,
        materialized_path=f"root/{current_section_id}",
    )
    ast.add_node(first_sec, parent_id=ast.root_id)

    para_counter = 1
    vis_counter = 1

    def _flush_paragraph(lines_buf: List[str], page_no: int, sec_id: str) -> Optional[ASTNode]:
        nonlocal para_counter
        joined = " ".join(ln.strip() for ln in lines_buf if ln.strip()).strip()
        if len(joined) <= 20:
            return None
        para_id = f"{doc_id}_para_{para_counter}"
        p_node = ASTNode(
            id=para_id,
            type=ASTNodeType.PARAGRAPH,
            title=f"Paragraph {para_counter} (Page {page_no})",
            text=joined,
            page=page_no,
            materialized_path=f"root/{sec_id}/{para_id}",
        )
        ast.add_node(p_node, parent_id=sec_id)
        para_counter += 1
        return p_node

    def _flush_table_visual(title_line: str, table_rows: List[str], page_no: int, sec_id: str) -> None:
        nonlocal vis_counter, para_counter
        vis_id = f"{doc_id}_vis_tbl_{vis_counter}"
        vis_counter += 1
        clean_title = title_line.strip().rstrip(":")
        full_table_text = "\n".join([title_line.strip()] + [r.strip() for r in table_rows])

        # Render a real WebP visual crop for text/markdown tables if not a PDF
        img_url = None
        if lazy_visuals and not pdf_bytes:
            img_url = f"/api/crop/{doc_id}/{vis_id}"
        elif not pdf_bytes:
            try:
                img_url = render_text_visual_crop(vis_id, clean_title, full_table_text)
            except Exception as e:
                print(f"Warning rendering text table crop: {e}")

        v_node = ASTNode(
            id=vis_id,
            type=ASTNodeType.VISUAL,
            title=clean_title,
            text=full_table_text,
            page=page_no,
            materialized_path=f"root/{sec_id}/{vis_id}",
            image_url=img_url,
            visual_type="table",
            confidence=0.95,
        )
        ast.add_node(v_node, parent_id=sec_id)

        # Also add a searchable paragraph node for the table so hybrid retrieval can match table cells directly
        para_id = f"{doc_id}_para_{para_counter}"
        p_node = ASTNode(
            id=para_id,
            type=ASTNodeType.PARAGRAPH,
            title=f"{clean_title} (Page {page_no})",
            text=full_table_text,
            page=page_no,
            materialized_path=f"root/{sec_id}/{para_id}",
            image_url=img_url,
            visual_type="table",
        )
        ast.add_node(p_node, parent_id=sec_id)
        _wire_bidirectional_link(p_node, v_node, tier=1, weight=1.0)
        para_counter += 1

    # 1. Parse pages into Sections, Paragraphs, and Structured Text Tables
    for page in pages:
        page_num = page.get("page", 1)
        raw_text = page.get("text", "").strip()
        lines = raw_text.splitlines()

        para_buffer: List[str] = []
        idx = 0
        while idx < len(lines):
            line_s = lines[idx].strip()

            # Blank line flushes current paragraph buffer
            if not line_s:
                if para_buffer:
                    _flush_paragraph(para_buffer, page_num, current_section_id)
                    para_buffer = []
                idx += 1
                continue

            # Ignore top-level document title (# Title)
            if line_s.startswith("# ") and not line_s.startswith("## "):
                idx += 1
                continue

            # Detect Section Header (## Heading or "1. UPPERCASE HEADING")
            is_sec_header = (
                line_s.startswith("## ")
                or bool(re.match(r'^\d+\.\s+[A-Z][A-Z0-9\s&/\-]{3,}$', line_s))
            )
            if is_sec_header:
                if para_buffer:
                    _flush_paragraph(para_buffer, page_num, current_section_id)
                    para_buffer = []
                clean_title = line_s.lstrip("#").strip()
                # If the initial default section has no children yet, replace it cleanly
                if sec_idx == 1 and not ast.nodes[current_section_id].children_ids:
                    ast.nodes[current_section_id].title = clean_title
                    ast.nodes[current_section_id].text = f"Section: {clean_title}"
                    ast.nodes[current_section_id].page = page_num
                    sec_idx += 1
                else:
                    sec_idx += 1
                    current_section_id = f"{doc_id}_sec_{sec_idx}"
                    sec_node = ASTNode(
                        id=current_section_id,
                        type=ASTNodeType.SECTION,
                        title=clean_title,
                        text=f"Section: {clean_title}",
                        page=page_num,
                        materialized_path=f"root/{current_section_id}",
                    )
                    ast.add_node(sec_node, parent_id=ast.root_id)
                idx += 1
                continue

            # Detect Table Header + Pipe Rows ("Table 1: ..." followed by "| ... |")
            is_table_header = bool(re.match(r'^(?:###\s*)?Table\s+\d+', line_s, re.IGNORECASE))
            if is_table_header and (idx + 1 < len(lines)) and lines[idx + 1].strip().startswith("|"):
                if para_buffer:
                    _flush_paragraph(para_buffer, page_num, current_section_id)
                    para_buffer = []
                tbl_title = line_s.lstrip("#").strip()
                tbl_rows = []
                idx += 1
                while idx < len(lines) and lines[idx].strip().startswith("|"):
                    tbl_rows.append(lines[idx].strip())
                    idx += 1
                _flush_table_visual(tbl_title, tbl_rows, page_num, current_section_id)
                continue

            # Detect standalone Pipe Table without explicit "Table X" title line
            if line_s.startswith("|") and line_s.endswith("|"):
                if para_buffer:
                    _flush_paragraph(para_buffer, page_num, current_section_id)
                    para_buffer = []
                tbl_rows = []
                while idx < len(lines) and lines[idx].strip().startswith("|"):
                    tbl_rows.append(lines[idx].strip())
                    idx += 1
                _flush_table_visual(f"Table {vis_counter} (Page {page_num})", tbl_rows, page_num, current_section_id)
                continue

            # Detect numbered sub-clause (e.g., "2.1 Uptime Guarantee:" or "### 2.1 ...")
            is_subclause = line_s.startswith("### ") or bool(re.match(r'^\d+\.\d+\s+', line_s))
            if is_subclause and para_buffer:
                _flush_paragraph(para_buffer, page_num, current_section_id)
                para_buffer = []

            clean_line = line_s[4:].strip() if line_s.startswith("### ") else line_s
            para_buffer.append(clean_line)
            idx += 1

        if para_buffer:
            _flush_paragraph(para_buffer, page_num, current_section_id)

    # 1.5 Synthesize Thematic Section Summaries (RAPTOR-Style AST Layer)
    for node in list(ast.nodes.values()):
        if node.type == ASTNodeType.SECTION:
            child_paras = [
                ast.nodes[cid] for cid in node.children_ids
                if cid in ast.nodes and ast.nodes[cid].type == ASTNodeType.PARAGRAPH
            ]
            if child_paras:
                lead_sentences = []
                for cp in child_paras[:4]:
                    clean = cp.text.split(". ")[0].strip()
                    if clean and len(clean) > 20 and not clean.startswith("Table") and not clean.startswith("|"):
                        lead_sentences.append(clean)
                if lead_sentences:
                    summary = "; ".join(lead_sentences)
                    if not summary.endswith("."):
                        summary += "."
                    node.metadata["thematic_summary"] = summary
                    node.metadata["summary_path"] = f"{node.materialized_path}/thematic_summary"
                    node.text = f"{node.title}: {summary}"

    # 2. Extract visual crops if PDF bytes are available
    if pdf_bytes:
        visual_nodes = extract_halo_crops(doc_id, pdf_bytes, render_crops=not lazy_visuals)
        # Attach each PDF visual node to the section active on its page (or root fallback)
        sections_by_page: Dict[int, str] = {}
        for n in ast.nodes.values():
            if n.type == ASTNodeType.SECTION:
                sections_by_page.setdefault(n.page, n.id)

        for v_node in visual_nodes:
            # Avoid duplicating a table visual if already extracted from text on the same page with same title
            duplicate = any(
                existing.type == ASTNodeType.VISUAL
                and existing.page == v_node.page
                and existing.title.lower() == v_node.title.lower()
                for existing in ast.nodes.values()
            )
            if duplicate:
                # Update existing visual node with the PDF halo-cropped image URL and bbox
                for existing in ast.nodes.values():
                    if (
                        existing.type == ASTNodeType.VISUAL
                        and existing.page == v_node.page
                        and existing.title.lower() == v_node.title.lower()
                    ):
                        existing.image_url = (
                            f"/api/crop/{doc_id}/{existing.id}" if lazy_visuals else v_node.image_url
                        )
                        existing.bbox = v_node.bbox
                continue

            parent_sec_id = sections_by_page.get(v_node.page, ast.root_id)
            v_node.materialized_path = f"root/{parent_sec_id}/{v_node.id}"
            ast.add_node(v_node, parent_id=parent_sec_id)

    # 3. Deterministic 3-Tier Cross-Modal Linking
    link_cross_modal_symbols(ast)

    return ast
