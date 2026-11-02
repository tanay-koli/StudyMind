"""StudyMind — structured PDF parsing (pages and section headings)."""

import pymupdf

from core.parse import Block, parse_pdf, text_blocks


def _pdf(tmp_path, pages):
    """pages: list of [(text, fontsize, fontname)] lines per page."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for text, size, font in lines:
            page.insert_text((72, y), text, fontsize=size, fontname=font)
            y += size * 2
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return str(path)


BODY = "Each token attends to every other token in the sequence using learned weights"


def test_headings_and_pages(tmp_path):
    pdf = _pdf(tmp_path, [
        [("Attention", 20, "helv"), (BODY, 11, "helv")],
        [("Transformers", 20, "helv"), (BODY, 11, "helv")],
    ])
    blocks = parse_pdf(pdf)
    assert {(b.page, b.section_heading) for b in blocks} == {(1, "Attention"), (2, "Transformers")}
    assert all(b.text.strip() == BODY for b in blocks)


def test_bold_short_line_is_heading(tmp_path):
    pdf = _pdf(tmp_path, [[("Key idea", 11, "hebo"), (BODY, 11, "helv")]])
    assert parse_pdf(pdf)[0].section_heading == "Key idea"


def test_heading_carries_to_next_page(tmp_path):
    pdf = _pdf(tmp_path, [
        [("Attention", 20, "helv"), (BODY, 11, "helv")],
        [(BODY, 11, "helv")],
    ])
    last = parse_pdf(pdf)[-1]
    assert (last.page, last.section_heading) == (2, "Attention")


def test_text_blocks():
    assert text_blocks("  ") == []
    assert text_blocks("hello") == [Block(None, None, "hello")]


def test_all_heading_pdf_keeps_its_text(tmp_path):
    pdf = _pdf(tmp_path, [[("Gradient descent", 11, "hebo"), ("Learning rate", 11, "hebo")]])
    assert [(b.page, b.text) for b in parse_pdf(pdf)] == [(1, "Gradient descent\nLearning rate")]


def test_consecutive_heading_lines_are_kept_together(tmp_path):
    pdf = _pdf(tmp_path, [[("Week 3", 20, "helv"), ("Key term: entropy", 11, "hebo"), (BODY, 11, "helv")]])
    assert parse_pdf(pdf)[0].section_heading == "Week 3\nKey term: entropy"


def test_table_becomes_one_markdown_block(tmp_path):
    from tests.pdfs import figure_pdf

    blocks = parse_pdf(str(figure_pdf(tmp_path / "f.pdf")))
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1 and tables[0].page == 1 and tables[0].section_heading == "Krebs Cycle"
    assert tables[0].text.splitlines() == [
        "| Enzyme | Rate | Step |", "| --- | --- | --- |", "| aconitase | 12 | 1 |", "| synthase | 8 | 2 |",
    ]
    prose = " ".join(b.text for b in blocks if b.kind == "text")
    assert "aconitase" not in prose and "Enzyme" not in prose
    assert "releases stored energy" in prose and "carbon dioxide" in prose


def _inside(inner, outer):
    return pymupdf.Rect(outer).contains(pymupdf.Rect(inner))


def test_extract_figures_finds_diagram_and_image_not_table(tmp_path):
    from core.parse import extract_figures
    from tests.pdfs import DIAGRAM_RECT, IMAGE_RECT, TABLE_RECT, figure_pdf

    crops = extract_figures(str(figure_pdf(tmp_path / "f.pdf")))
    assert [c.page for c in crops] == [1, 1]  # page 2 is only a background
    boxes = [pymupdf.Rect(c.bbox) for c in crops]
    assert any(_inside(DIAGRAM_RECT, b) for b in boxes) and any(_inside(IMAGE_RECT, b) for b in boxes)
    assert not any(b.intersects(TABLE_RECT) for b in boxes)
    assert not any(b.contains(pymupdf.Point(515, 795)) for b in boxes)  # the 30 pt logo
    assert all(c.png.startswith(b"\x89PNG") for c in crops)


def test_scanned_page_queues_no_figures(tmp_path):
    from core.parse import extract_figures
    from tests.pdfs import scanned_pdf

    path = str(scanned_pdf(tmp_path / "s.pdf"))
    assert extract_figures(path) == [] and parse_pdf(path) == []


def test_extract_figures_survives_a_bad_page(tmp_path, monkeypatch):
    from core.parse import extract_figures
    from tests.pdfs import diagram_only_pdf, figure_pdf

    path = tmp_path / "two.pdf"
    doc = pymupdf.open(figure_pdf(tmp_path / "f.pdf"))
    doc.insert_pdf(pymupdf.open(diagram_only_pdf(tmp_path / "d.pdf")))
    doc.save(path)  # pages: figures, background, prose, diagram
    calls = {"n": 0}
    real = pymupdf.Page.cluster_drawings

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("bad page")
        return real(self, *a, **kw)

    monkeypatch.setattr(pymupdf.Page, "cluster_drawings", flaky)
    assert [c.page for c in extract_figures(str(path))] == [4]
