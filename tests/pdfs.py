"""Synthetic PDFs for the figure and table tests (A4: 595 x 842 pt)."""

from pathlib import Path

import pymupdf

TABLE_RECT = pymupdf.Rect(72, 130, 372, 220)
DIAGRAM_RECT = pymupdf.Rect(100, 260, 480, 360)
IMAGE_RECT = pymupdf.Rect(100, 420, 250, 570)


def png(width: int = 64, height: int = 64, gray: int = 200) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 0)
    pix.clear_with(gray)
    return pix.tobytes("png")


def _diagram(page, left: str = "citrate", right: str = "isocitrate") -> None:
    """Two labelled boxes joined by an arrow line, drawn as vector shapes."""
    a = pymupdf.Rect(DIAGRAM_RECT.x0, DIAGRAM_RECT.y0, DIAGRAM_RECT.x0 + 150, DIAGRAM_RECT.y1)
    b = pymupdf.Rect(DIAGRAM_RECT.x1 - 150, DIAGRAM_RECT.y0, DIAGRAM_RECT.x1, DIAGRAM_RECT.y1)
    page.draw_rect(a, color=(0, 0, 0), width=1.5)
    page.draw_rect(b, color=(0, 0, 0), width=1.5)
    mid = (a.y0 + a.y1) / 2
    page.draw_line((a.x1, mid), (b.x0, mid), color=(0, 0, 0), width=1.5)
    page.insert_text((a.x0 + 20, mid), left, fontsize=11)
    page.insert_text((b.x0 + 20, mid), right, fontsize=11)


def _table(page) -> None:
    """A ruled 3 x 3 table."""
    cells = [["Enzyme", "Rate", "Step"], ["aconitase", "12", "1"], ["synthase", "8", "2"]]
    x0, y0, w, h = TABLE_RECT.x0, TABLE_RECT.y0, 100, 30
    for r in range(4):
        page.draw_line((x0, y0 + r * h), (x0 + 3 * w, y0 + r * h), color=(0, 0, 0), width=1)
    for c in range(4):
        page.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + 3 * h), color=(0, 0, 0), width=1)
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page.insert_text((x0 + c * w + 6, y0 + r * h + 20), text, fontsize=10)


def figure_pdf(path: Path) -> Path:
    """Page 1: heading, prose, table, vector diagram, embedded image, tiny logo. Page 2: full-page background."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Krebs Cycle", fontsize=20)
    page.insert_text((72, 100), "The citric acid cycle releases stored energy in every cell.", fontsize=11)
    _table(page)
    _diagram(page)
    page.insert_image(IMAGE_RECT, stream=png())
    page.draw_rect(pymupdf.Rect(500, 780, 530, 810), color=(0, 0, 0), fill=(0.2, 0.2, 0.8))
    page.insert_text((72, 620), "Each turn of the cycle produces carbon dioxide.", fontsize=11)

    page = doc.new_page()
    page.draw_rect(page.rect, color=None, fill=(0.95, 0.95, 0.9))
    page.insert_text((72, 72), "Appendix notes for revision.", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def diagram_only_pdf(path: Path) -> Path:
    """Page 1: unrelated prose. Page 2: a heading and a diagram whose labels appear nowhere else."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Photosynthesis", fontsize=20)
    page.insert_text((72, 100), "Plants capture light energy in chloroplasts to make glucose.", fontsize=11)
    page = doc.new_page()
    page.insert_text((72, 60), "Cycle overview", fontsize=20)
    _diagram(page)
    doc.save(path)
    doc.close()
    return path


def scanned_pdf(path: Path) -> Path:
    """One page that is a single full-page image (page-shaped, as a scan is), no text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=png(595, 842, 240))
    doc.save(path)
    doc.close()
    return path
