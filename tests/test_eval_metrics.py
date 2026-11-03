"""StudyMind — eval metrics: gold matching, Recall@k, MRR, citation accuracy, kappa."""

from eval import metrics as m


def T(title, page, text, kind="text"):
    return {"source_title": title, "page": page, "text": text, "kind": kind}


Q = {"source": "ODS", "page": 3, "quote": "A skiplist is a sequence"}
F = {"source": "ODS", "page": 9, "kind": "figure"}


def test_normalise_joins_hyphen_breaks_and_whitespace():
    assert m.normalise("Skip-\nList  is\n a  LIST") == "skiplist is a list"


def test_quote_match_needs_same_source_and_contained_quote():
    assert m.matches(T("ODS", 4, "Intro. A  skiplist is a\nsequence of lists."), Q)  # page not required for quotes
    assert not m.matches(T("Other", 3, "A skiplist is a sequence"), Q)
    assert not m.matches(T("ODS", 3, "A skiplist is not a sequence"), Q)


def test_figure_gold_matches_figure_hit_on_page_only():
    assert m.matches(T("ODS", 9, "any caption", "figure"), F)
    assert not m.matches(T("ODS", 9, "any caption", "text"), F)
    assert not m.matches(T("ODS", 8, "any caption", "figure"), F)


def test_recall_and_mrr():
    hits = [T("ODS", 1, "x"), T("ODS", 9, "c", "figure"), T("ODS", 3, "a skiplist is a sequence of")]
    assert m.recall_at(hits, [Q, F]) == 1.0
    assert m.recall_at(hits, [Q, F], k=2) == 0.5
    assert m.mrr(hits, [Q, F]) == 0.5
    assert m.mrr([T("ODS", 1, "x")], [Q]) == 0.0
    assert m.recall_at([], [Q]) == 0.0


def test_citation_accuracy():
    hits = [T("ODS", 3, "a skiplist is a sequence"), T("ODS", 1, "x")]
    assert m.citation_accuracy("Yes [1] and [2].", hits, [Q]) == 0.5
    assert m.citation_accuracy("Yes [7].", hits, [Q]) == 0.0
    assert m.citation_accuracy("No citations.", hits, [Q]) is None


def test_kappa():
    assert m.cohen_kappa(["c", "w", "c", "w"], ["c", "w", "c", "w"]) == 1.0
    assert round(m.cohen_kappa(["c", "c", "w", "w"], ["c", "w", "c", "w"]), 3) == 0.0
    assert m.cohen_kappa(["c", "c"], ["c", "c"]) == 1.0


def test_is_refusal():
    assert m.is_refusal("  This isn't in your material. Searched for: 'x'")
    assert not m.is_refusal("It is in your material [1].")
