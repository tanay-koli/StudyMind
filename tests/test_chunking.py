"""StudyMind — chunkers (whitespace tokens via the conftest fake)."""

from core.chunking import fixed, structure
from core.parse import Block


def words(n, start=0):
    return " ".join(f"w{i}" for i in range(start, start + n))


def test_fixed_window_and_overlap():
    out = fixed([Block(1, "A", words(1000))], size=512, overlap=64)
    assert [len(c.text.split()) for c in out] == [512, 512, 104]
    assert out[1].text.split()[0] == "w448"


def test_fixed_takes_page_of_start_block():
    out = fixed([Block(1, "A", words(500)), Block(2, "B", words(500, 500))])
    # The second window starts at token 448, still inside the first block
    assert [(c.page, c.section_heading) for c in out] == [(1, "A"), (1, "A"), (2, "B")]


def test_structure_splits_on_heading_change():
    out = structure([Block(1, "A", "a1"), Block(1, "A", "a2"), Block(2, "B", "b1")])
    assert [(c.page, c.section_heading, c.text) for c in out] == [(1, "A", "a1\n\na2"), (2, "B", "b1")]


def test_structure_packs_up_to_max_tokens():
    out = structure([Block(1, "A", words(300)), Block(2, "A", words(300, 300))])
    assert [(c.page, len(c.text.split())) for c in out] == [(1, 300), (2, 300)]


def test_structure_splits_oversize_block():
    out = structure([Block(3, "A", words(1200))])
    assert len(out) == 3
    assert all(len(c.text.split()) <= 512 and c.page == 3 and c.section_heading == "A" for c in out)


def test_tiny_and_blank_inputs():
    assert [c.text for c in fixed([Block(1, None, "hi")])] == ["hi"]
    assert fixed([Block(1, None, "   ")]) == []
    assert structure([]) == []
    assert structure([Block(1, "A", "  ")]) == []


def test_structure_never_crosses_pages():
    # A chunk cites one page, so a section spanning pages splits at the page break
    out = structure([Block(1, "A", "end of page one"), Block(2, "A", "start of page two")])
    assert [(c.page, c.text) for c in out] == [(1, "end of page one"), (2, "start of page two")]
