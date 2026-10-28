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
